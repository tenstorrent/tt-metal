# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import signal
import shutil
import subprocess
import time
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger


class ResetFailed(RuntimeError):
    """Raised when ALL configured reset mechanisms are exhausted without success.

    Signals that the device is unrecoverable on this host, so the caller should
    abort rather than keep launching vectors against a wedged device.
    """


class ResetUtil:
    SUPPORTED_ARCHS = {"wormhole_b0", "blackhole"}

    def __init__(self, arch: str):
        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")

        self.arch = arch
        # Ordered list of reset mechanisms to try; a second, DIFFERENT mechanism
        # is attempted if the first is exhausted (see _find_commands).
        self.mechanisms = self._find_commands()
        # Back-compat: expose the primary mechanism as command/args.
        self.command, self.args = self.mechanisms[0]
        # Retry policy (overridable via env for tuning per runner).
        self.reset_attempts = max(1, int(os.getenv("TT_SMI_RESET_ATTEMPTS", "3")))
        self.reset_backoff_seconds = max(0, int(os.getenv("TT_SMI_RESET_BACKOFF_SECONDS", "30")))
        # After a reset, wait for the kernel driver / UMD to finish re-enumerating
        # the devices before returning. A PCIe-level reset (tt-smi -r) tears the
        # driver down; without a settle the next process can see <N devices or hit
        # "Cannot access soc descriptor ... before device driver is initialized".
        self.post_reset_settle_seconds = max(0, int(os.getenv("TT_SMI_POST_RESET_SETTLE_SECONDS", "10")))

    def _find_commands(self):
        """Build the ordered list of reset mechanisms [(executable, args), ...].

        The galaxy IPMI/tray reset (``tt-smi -glx_reset*``) and the PCIe-level
        reset (``tt-smi -r``) fail independently: e.g. the tray reset can fail its
        POST_RESET on a single wedged device while the PCIe reset still recovers
        it (and vice-versa). So when the primary mechanism is exhausted we fall
        back to the *other* mechanism before giving up.

        Primary:  TT_SMI_RESET_COMMAND, else ``tt-smi -r``.
        Fallback: TT_SMI_RESET_FALLBACK_COMMAND, else auto-picked as the opposite
                  mechanism to the primary. Set the fallback env to empty/``none``
                  to disable.
        """
        tt_smi = shutil.which("tt-smi")

        def _parse(cmd_str):
            parts = cmd_str.split()
            if not shutil.which(parts[0]):
                raise FileNotFoundError(f"SWEEPS: reset command not found: {parts[0]}")
            return (shutil.which(parts[0]), parts[1:])

        primary_env = os.getenv("TT_SMI_RESET_COMMAND")
        if primary_env:
            primary = _parse(primary_env)
        else:
            if not tt_smi:
                raise FileNotFoundError("SWEEPS: Unable to locate tt-smi executable")
            logger.info(f"tt-smi executable: {tt_smi}")
            primary = (tt_smi, ["-r"])

        mechanisms = [primary]

        fallback_env = os.getenv("TT_SMI_RESET_FALLBACK_COMMAND")
        if fallback_env is not None:
            if fallback_env.strip() and fallback_env.strip().lower() != "none":
                mechanisms.append(_parse(fallback_env))
        elif tt_smi:
            p_args = primary[1]
            if any("glx" in a for a in p_args):
                mechanisms.append((tt_smi, ["-r", "all"]))  # PCIe-level reset
            elif any(a in ("-r", "--reset") for a in p_args):
                mechanisms.append((tt_smi, ["-glx_reset"]))  # IPMI/tray reset

        # De-duplicate identical mechanisms (keep order).
        seen, uniq = set(), []
        for exe, args in mechanisms:
            key = (exe, tuple(args))
            if key not in seen:
                seen.add(key)
                uniq.append((exe, args))
        logger.info("SWEEPS: reset mechanisms: " + ", ".join("'" + " ".join([e, *a]) + "'" for e, a in uniq))
        return uniq

    def _self_and_ancestors(self):
        """PIDs of this process and its ancestor chain (never to be killed)."""
        pids = set()
        pid = os.getpid()
        while pid and pid > 1 and pid not in pids:
            pids.add(pid)
            try:
                with open(f"/proc/{pid}/stat") as f:
                    # ppid is the 4th field, but the comm (2nd field) may contain
                    # spaces/parens — parse after the closing ')'.
                    data = f.read()
                    pid = int(data[data.rindex(")") + 1 :].split()[1])
            except Exception:
                break
        return pids

    def _device_holder_pids(self):
        """PIDs (excluding self + ancestors) holding a /dev/tenstorrent fd.

        A vector that hangs in a device call is SIGKILLed by the runner before
        the reset, but a process stuck in an uninterruptible (D-state) driver
        call keeps the device claimed until that call returns; tt-smi then can't
        reset it and returns non-zero. Best-effort kill of such leftovers (and
        a short wait) lets the reset proceed.
        """
        protected = self._self_and_ancestors()
        holders = []
        try:
            proc_entries = os.listdir("/proc")
        except OSError:
            return holders
        for entry in proc_entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in protected:
                continue
            fd_dir = f"/proc/{entry}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if "tenstorrent" in target:
                    holders.append(pid)
                    break
        return holders

    def _free_device(self):
        """Best-effort: SIGKILL stray processes still holding the device."""
        holders = self._device_holder_pids()
        if not holders:
            return
        logger.warning(f"SWEEPS: killing stray process(es) holding /dev/tenstorrent before reset: {holders}")
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                # pid already gone or not ours (race / permission) — reset proceeds regardless
                pass
        # Give the kernel a moment to tear down the killed processes' handles.
        time.sleep(2)

    def reset(self):
        """Reset the device, trying each mechanism with cleanup + backoff retries.

        glx_reset_auto / tt-smi -r can fail (exit 1) when the device is still
        claimed by a just-killed-but-not-yet-reaped hang, or is transiently busy.
        Free the device first and retry a few times with a backoff so it has time
        to be released. If the primary mechanism stays failed after all attempts,
        fall back to the OTHER reset mechanism (PCIe vs IPMI/tray) before giving
        up — they fail independently, so the fallback often recovers a device the
        primary cannot. Raises ResetFailed only when every mechanism is exhausted.
        """
        last_rc = None
        n_mech = len(self.mechanisms)
        for mech_idx, (command, args) in enumerate(self.mechanisms, 1):
            label = " ".join([os.path.basename(command), *args])
            for attempt in range(1, self.reset_attempts + 1):
                try:
                    self._free_device()
                except Exception as e:
                    logger.warning(f"SWEEPS: device-holder cleanup failed (continuing to reset): {e}")

                # Surface tt-smi output on the final attempt of the final mechanism.
                show_output = mech_idx == n_mech and attempt == self.reset_attempts
                # Suppress BOTH streams on non-final attempts — tt-smi reports failures
                # largely on stderr, so silencing only stdout still spams intermediate retries.
                result = subprocess.run(
                    [command, *args],
                    stdout=None if show_output else subprocess.DEVNULL,
                    stderr=None if show_output else subprocess.DEVNULL,
                )
                last_rc = result.returncode
                if last_rc == 0:
                    logger.info(
                        f"TT-SMI Reset Complete Successfully via '{label}' (attempt {attempt}/{self.reset_attempts})"
                    )
                    # Let the kernel driver / UMD re-enumerate before the next op
                    # opens a device (esp. after a PCIe '-r' reset, which tears the
                    # driver down). Without this, the next process can transiently
                    # see <N devices and skip/fail with "soc descriptor ... before
                    # device driver is initialized".
                    if self.post_reset_settle_seconds:
                        logger.info(
                            f"SWEEPS: waiting {self.post_reset_settle_seconds}s for device driver re-enumeration "
                            f"after '{label}'."
                        )
                        time.sleep(self.post_reset_settle_seconds)
                    return
                if attempt < self.reset_attempts:
                    logger.warning(
                        f"SWEEPS: reset '{label}' attempt {attempt}/{self.reset_attempts} failed (exit {last_rc}); "
                        f"waiting {self.reset_backoff_seconds}s for the device to be released, then retrying."
                    )
                    time.sleep(self.reset_backoff_seconds)
            if mech_idx < n_mech:
                logger.warning(
                    f"SWEEPS: reset mechanism '{label}' exhausted after {self.reset_attempts} attempts "
                    f"(exit {last_rc}); falling back to the next reset mechanism."
                )

        raise ResetFailed(
            f"SWEEPS: TT-SMI Reset Failed with Exit Code: {last_rc} after exhausting all "
            f"{n_mech} reset mechanism(s) × {self.reset_attempts} attempts"
        )
