#!/usr/bin/env bash
# Throwaway probe: can this process open a Tenstorrent card?
#
# Run once outside the gh-aw agent sandbox (control) and once inside it
# (treatment). The difference tells us whether AWF's container blocks device
# access, which decides whether an agentic porting workflow can measure on a
# card in-loop or has to dispatch a separate device job.
#
# Never exits non-zero: a denied open is a result, not a failure.

LABEL="${1:-unlabelled}"
echo "================ AWF DEVICE PROBE [${LABEL}] ================"

echo "--- identity ---"
id
echo "hostname: $(hostname)"

echo "--- containerised? ---"
if [ -f /.dockerenv ]; then echo "/.dockerenv present -> inside a container"; else echo "/.dockerenv absent"; fi
echo "cgroup: $(head -1 /proc/self/cgroup 2>/dev/null || echo unknown)"
echo "AWF env: $(env | grep -c '^AWF' 2>/dev/null) AWF_* vars"

echo "--- /dev/tenstorrent ---"
ls -la /dev/tenstorrent/ 2>&1 || echo "MISSING /dev/tenstorrent"

echo "--- hugepages ---"
ls -d /dev/hugepages-1G 2>&1 || echo "MISSING /dev/hugepages-1G"

echo "--- open() probe ---"
python3 - <<'PY'
import errno, glob, os, stat

nodes = sorted(glob.glob("/dev/tenstorrent/*"))
print(f"device nodes found: {nodes or 'NONE'}")
if not nodes:
    print("VERDICT: no device nodes visible")
    raise SystemExit(0)

verdicts = []
for n in nodes:
    try:
        st = os.stat(n)
        kind = "chardev" if stat.S_ISCHR(st.st_mode) else oct(st.st_mode)
        print(f"  {n}: mode={oct(st.st_mode & 0o777)} type={kind} "
              f"rdev={os.major(st.st_rdev)}:{os.minor(st.st_rdev)}")
    except OSError as e:
        print(f"  {n}: stat failed {e}")
    for flag, name in ((os.O_RDWR, "O_RDWR"), (os.O_RDONLY, "O_RDONLY")):
        try:
            fd = os.open(n, flag)
            os.close(fd)
            print(f"    OPEN OK   {name}")
            verdicts.append(("ok", n, name))
        except OSError as e:
            code = errno.errorcode.get(e.errno, "?")
            print(f"    OPEN FAIL {name} errno={e.errno} ({code}) {e.strerror}")
            verdicts.append((code, n, name))

if any(v[0] == "ok" for v in verdicts):
    print("VERDICT: device is openable from this context")
elif any(v[0] == "EPERM" for v in verdicts):
    print("VERDICT: EPERM -> blocked by the container device cgroup (no --device rule)")
elif any(v[0] == "EACCES" for v in verdicts):
    print("VERDICT: EACCES -> blocked by file permissions / group membership")
else:
    print("VERDICT: blocked, see errno above")
PY

echo "--- tt-smi / umd present? ---"
command -v tt-smi >/dev/null 2>&1 && echo "tt-smi: $(command -v tt-smi)" || echo "tt-smi: not on PATH"

echo "================ END PROBE [${LABEL}] ================"
exit 0
