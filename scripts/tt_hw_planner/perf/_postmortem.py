# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Post-mortem for failed `perf collect` runs.

When a bring-up crashes inside the demo, this module reads `collect.log`,
extracts the Python `AssertionError` (or other fatal traceback signal),
and turns it into an actionable suggestion the USER can apply without
asking the assistant.

The goal is to convert each new failure into a small, self-service patch
to one of the planner registries (`kernel_constraints.py`,
`compatibility.py`, `model_targets.yaml`) instead of a back-and-forth
debugging session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PostMortem:
    """Structured summary of what crashed and how to teach the planner.

    Three layers, in increasing actionability:

      matched_pattern + assertion_text + suggested_action
            (human-readable; what we've always had)

      quantified
            (machine-readable evidence parsed from the log: which devices,
             how many overflows, what cap was hit, etc. — the part that lets
             the framework reason about the failure instead of guessing)

      next_retry_args + retry_explanation
            (a concrete kwargs dict the CLI can splice into the next
             `collect_run` call. Non-None means the failure is auto-recoverable;
             the CLI's auto-retry loop will use it. None means human action
             required.)
    """

    matched_pattern: Optional[str]  # human-readable name of the known signature, or None
    assertion_text: Optional[str]
    file_line: Optional[Tuple[str, int]]
    traceback_tail: List[str] = field(default_factory=list)
    suggested_action: List[str] = field(default_factory=list)
    quantified: Dict[str, Any] = field(default_factory=dict)
    next_retry_args: Optional[Dict[str, Any]] = None
    retry_explanation: Optional[str] = None

    def render(self) -> str:
        out: List[str] = []
        out.append("=" * 70)
        out.append("POST-MORTEM (auto-generated from collect.log)")
        out.append("=" * 70)
        if self.matched_pattern is not None:
            out.append(f"  Diagnosed: {self.matched_pattern}")
        else:
            out.append("  Diagnosed: UNKNOWN constraint pattern")
            out.append("  This assertion is not yet in the planner registry. Adding")
            out.append("  it is the right fix; the steps below show exactly what to")
            out.append("  edit so future runs of any model with the same constraint")
            out.append("  are caught pre-flight.")
        if self.assertion_text:
            out.append("")
            out.append(f"  Assertion: {self.assertion_text}")
        if self.file_line:
            out.append(f"  At:        {self.file_line[0]}:{self.file_line[1]}")

        if self.quantified:
            out.append("")
            out.append("  Quantified evidence (parsed from log):")
            for k, v in self.quantified.items():
                # Render lists/sets compactly; longer values get wrapped onto their own line.
                if isinstance(v, (list, tuple)) and len(v) > 6:
                    head = ", ".join(str(x) for x in list(v)[:6])
                    out.append(f"    {k:>28s} : [{head}, ...] (total {len(v)})")
                else:
                    out.append(f"    {k:>28s} : {v}")

        if self.next_retry_args is not None:
            out.append("")
            out.append("  Auto-recoverable: the framework can retry this run with:")
            for k, v in self.next_retry_args.items():
                out.append(f"    {k} = {v}")
            if self.retry_explanation:
                out.append(f"    rationale: {self.retry_explanation}")
            out.append("  Use `--no-auto-retry` to disable the auto-retry loop.")

        if self.suggested_action:
            out.append("")
            out.append("  Suggested action:")
            for ln in self.suggested_action:
                out.append(f"    {ln}")
        if self.traceback_tail:
            out.append("")
            out.append("  Last frames of the traceback (for context):")
            for ln in self.traceback_tail[-12:]:
                out.append(f"    {ln.rstrip()}")
        out.append("=" * 70)
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Known patterns: things we already know how to teach the planner.
# Add new entries here as you discover new failure modes.
# ---------------------------------------------------------------------------


@dataclass
class _Pattern:
    """A regex that matches a known assert + an explanation."""

    name: str
    regex: re.Pattern
    suggestion: List[str]


_FABRIC_SYNC_TIMEOUT_PATTERN = _Pattern(
    # Multi-chip fabric (ethernet) handshake failed at device-init. This
    # asserts in TT_THROW before any model code runs, so 0 device-kernel
    # rows land in the Tracy CSV. The cause is environmental: a chip is
    # held in a bad state by a prior crash, a link is mistrained, or a
    # cable/firmware regression. NOT a model constraint — there is no
    # check to add to kernel_constraints.py.
    name="Fabric router sync timeout (multi-chip ethernet handshake)",
    regex=re.compile(
        r"Fabric Router Sync:\s*Timeout" r"|fabric_firmware_initializer\.cpp" r"|Ethernet handshake likely failed"
    ),
    suggestion=[
        "DEVICE INIT failed: the fabric router did not complete its ethernet",
        "  handshake within 10s. This fires BEFORE any model op runs (0 device-",
        "  kernel rows in the Tracy CSV). Model-agnostic.",
        "",
        "  IMPORTANT: `simple_text_demo.py` hard-codes `fabric_config=FABRIC_1D`",
        "  in its `device_params` parametrize, so this fires even at mesh=[1,1].",
        "  The bisect below is therefore NOT a 'skip fabric' test — it's a",
        "  'minimise fabric scope' test. If [1,1] also times out, the box's",
        "  device init is broken for every mesh, not specifically multi-chip.",
        "",
        "  This is NOT a constraint to add to kernel_constraints.py. No static",
        "  predicate decides whether two BH chips will handshake.",
        "",
        "  Recovery (cheapest first):",
        "    1. Reset the chips:                  tt-smi -r",
        "       (no args = reset every chip the host sees.)",
        "    2. Re-run perf collect.              Same command works after reset.",
        "    3. Auto-bisect to mesh=[1,1].        On retry, `perf collect` shrinks",
        "       to the smallest possible mesh. If that ALSO times out, fabric",
        "       init is broken for every mesh — the box, not the link topology,",
        "       is the problem.",
        "    4. Inspect link state in telemetry:  tt-smi -s --snapshot_no_tty | rg ETH_LIVE_STATUS",
        "       (ETH_LIVE_STATUS=0x0 on any chip => the ethernet endpoint is not",
        "        trained live; no software flag in this repo can fix that.)",
        "    5. If reset + bisect both fail with bad link state, this is a",
        "       firmware/driver issue — file against tt_metal with the postmortem",
        "       JSON (postmortem.txt + quantified evidence below).",
    ],
)


def _recommend_retry_for_fabric_timeout(
    ev: Dict[str, Any],
    prev_ev: Optional[Dict[str, Any]] = None,
    *,
    current_mesh: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Translate fabric router timeout evidence into a concrete retry.

    Strategy: auto-bisect once to ``mesh=(1, 1)``. A single-chip mesh skips
    fabric init entirely. The bisect outcome is dispositive:

      - succeeds  -> the model is fine; the multi-chip ethernet link is the
        problem; user should reset / file a hardware bug.
      - fails too -> the failure is NOT multi-chip-fabric-specific; rerun
        post-mortem on the second log to identify what's actually broken.

    Bisect once: if we are already at (1, 1) when this is called (because a
    previous retry brought us here), there's nothing left to bisect.
    """
    # Already at single-chip mesh — bisect can't help further.
    if current_mesh is not None and tuple(current_mesh) == (1, 1):
        # On a single-chip mesh, fabric_config=FABRIC_1D is still set by the
        # demo's parametrize, so fabric init still runs but with the smallest
        # possible scope. Timing out HERE means the box's fabric/eth init
        # subsystem is unhealthy for EVERY mesh size, not specifically the
        # multi-chip link. There is nothing in this repo that can recover it.
        stuck = ev.get("stuck_device", "?")
        return None, (
            f"fabric router sync timed out even at mesh=[1,1] (device {stuck} "
            f"stuck at {ev.get('earliest_init_stage_reached')}). Single-chip "
            "still triggers fabric init because the demo's device_params hard-"
            "codes fabric_config=FABRIC_1D, but the SCOPE is minimised — and "
            "it still fails. Conclusion: this box's device init is broken for "
            "every mesh, not just multi-chip. The link/firmware/driver needs "
            "attention from tt_metal. No software flag in this repo can fix it."
        )

    # Stuck across retries -> reset isn't helping; bisect is the right move.
    if prev_ev is not None and _fabric_evidence_essentially_equal(ev, prev_ev):
        return {"mesh_override": (1, 1)}, (
            "identical fabric router timeout across attempts (Device "
            f"{ev.get('stuck_device')} stuck at "
            f"{ev.get('earliest_init_stage_reached')}); auto-bisecting to "
            "mesh=[1,1] to determine whether the model itself runs."
        )

    # First attempt: still try the bisect now rather than failing twice.
    # The user has already been told (by the suggestion text) to `tt-smi -r`
    # manually; auto-retrying with no reset would be a no-op. Bisecting
    # directly is the only retry that produces new information.
    return {"mesh_override": (1, 1)}, (
        "fabric router sync timeout at multi-chip mesh; auto-bisecting to "
        "mesh=[1,1] (single chip skips fabric init) to determine whether "
        "the model itself works on this hardware."
    )


def _fabric_evidence_essentially_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Two fabric timeouts are 'the same' when the same device gets stuck at
    the same init stage. Differences in timing or which sub-channels are
    listed don't count."""
    return a.get("stuck_device") == b.get("stuck_device") and a.get("earliest_init_stage_reached") == b.get(
        "earliest_init_stage_reached"
    )


_BUFFER_OVERFLOW_PATTERN = _Pattern(
    # The root cause of `Device data missing` on multi-device runs is a
    # Metal-layer DRAM ring overflow that logs as
    #   "Profiler DRAM buffers were full, markers were dropped! device N,
    #    worker core X, Y, Risc BRISC, bufferEndIndex = 12000."
    # The downstream tracy assert is a SYMPTOM, not the cause. We match this
    # pattern by sniffing the full log text in `analyze_log`, not via
    # `_match_known_pattern` (which only sees the parsed assertion text).
    name="Metal device profiler DRAM ring overflow (bufferEndIndex=12000)",
    regex=re.compile(  # kept for documentation; sniffer uses _OVERFLOW_LINE_RX
        r"Profiler DRAM buffers were full, markers were dropped"
        r"|Device data missing: Op \d+ not present in .*device_perf_report"
    ),
    suggestion=[
        "The Metal device-side profiler ring buffer (12000 markers per",
        "  (device, core, RISC)) overflowed and dropped markers. Tracy's",
        "  post-processing then asserts on the missing rows.",
        "",
        "  `perf collect` already runs with `--dump-device-data-mid-run` and",
        "  `TT_METAL_PROFILER_MID_RUN_DUMP=1`. If those are on and you still",
        "  overflow, the only knob is the run length: lower",
        "  `--max-generated-tokens`. The framework auto-halves it on retry.",
        "",
        "  Model-agnostic: the ring is per-device, not per-model.",
    ],
)


_KNOWN_PATTERNS: List[_Pattern] = [
    _Pattern(
        name="n_kv_heads % cluster_shape[1] != 0",
        regex=re.compile(r"n_kv_heads must be divisible by num_devices"),
        suggestion=[
            "Already in kernel_constraints.py (num_key_value_heads divisibility).",
            "If you still hit this, run `tt_hw_planner compat <MODEL>` and check the",
            "  `Per-TP divisibility` table at the chosen mesh's TP. If TP is shown as",
            "  [ ok ] for TP=N but the runtime asserts at N, the model config is",
            "  reporting different head counts than the planner saw — clear the HF",
            "  config cache and retry.",
        ],
    ),
    _Pattern(
        name="n_heads % cluster_shape[1] != 0",
        regex=re.compile(r"n_heads must be divisible by num_devices"),
        suggestion=[
            "Already in kernel_constraints.py (num_attention_heads divisibility).",
            "Pick a mesh whose TP divides n_heads (often 1, 2, 4, 8).",
        ],
    ),
    _Pattern(
        name="hidden_size % TP != 0",
        regex=re.compile(r"hidden_size must be divisible by TP"),
        suggestion=[
            "Already in kernel_constraints.py (hidden_size divisibility).",
            "Pick a mesh whose TP divides hidden_size.",
        ],
    ),
    _Pattern(
        name="head_dim not multiple of TILE",
        regex=re.compile(r"head_dim must be a multiple of TILE"),
        suggestion=[
            "Already in kernel_constraints.py.",
            "This model needs a TT-supported sibling with aligned head_dim.",
        ],
    ),
    _Pattern(
        name="NOC_MAX_BURST_SIZE static_assert (Wormhole-only kernel on Blackhole)",
        regex=re.compile(r"coalesced_read_bytes\s*<=\s*NOC_MAX_BURST_SIZE"),
        suggestion=[
            "The demo uses a kernel ported only to Wormhole. The planner's arch",
            "  compat-gate should refuse this — check that the demo path is listed",
            "  in `discovery.arch_compatibility` for Blackhole.",
            "Edit discovery.py and add the demo path to the WORMHOLE_ONLY set if",
            "  this is in fact a Wormhole-only demo; the gate in bringup.py will",
            "  then refuse pre-flight.",
        ],
    ),
    _Pattern(
        name="capped_warmup_seq_len must be a power of 2",
        regex=re.compile(r"capped_warmup_seq_len must be a power of 2"),
        suggestion=[
            "Warmup-seq-len table is missing or wrong for this model on this SKU.",
            "Add an entry to MAX_PREFILL_CHUNK_SIZES_DIV1024 in",
            "  models/tt_transformers/tt/model_config.py (or rely on",
            "  `tt_hw_planner scaffold --apply` to insert a default).",
        ],
    ),
    _Pattern(
        name="dram-sharded matmul k % (TILE * num_cores) != 0",
        regex=re.compile(r"k must be divisible by tile_size \* num_cores"),
        suggestion=[
            "This is a per-op constraint that fires during graph construction.",
            "It is NOT yet in kernel_constraints.py. Add a check like:",
            "",
            "    if h is not None and tp > 1:",
            "        for grid_x in (1, 2, 4, 8):",
            "            num_cores = grid_x * tp",
            "            k_per_chip = (h // tp)",
            "            out.append(KernelFinding(",
            "                op='ttnn.matmul (dram-sharded)',",
            "                field='k',  value=k_per_chip,",
            "                constraint=f'k must be divisible by TILE*num_cores={32*num_cores}',",
            "                passes=(k_per_chip % (32*num_cores) == 0),",
            "                severity=Severity.BLOCKER,",
            "                fix='Pick mesh/grid so k is divisible by 32*num_cores.',",
            "                source='models/tt_transformers/tt/model_config.py L3281',",
            "            ))",
        ],
    ),
    _Pattern(
        name="Unsupported model on device",
        regex=re.compile(r"Unsupported model .* on device"),
        suggestion=[
            "MAX_PREFILL_CHUNK_SIZES_DIV1024 is missing this (model, device) pair.",
            "Run `tt_hw_planner scaffold <MODEL> --apply` to insert the default row,",
            "  or edit models/tt_transformers/tt/model_config.py directly.",
        ],
    ),
    _Pattern(
        name="L1 / DRAM out-of-memory at allocation",
        regex=re.compile(
            r"(Out of Memory: Not enough space to allocate"
            r"|allocate_buffer.*failed"
            r"|Cannot allocate \d+ bytes"
            r"|TT_THROW.*memory)"
        ),
        suggestion=[
            "The model does not fit at the chosen (box, mesh, dtype).",
            "Re-run `tt_hw_planner plan <MODEL> --box <BOX> --all-meshes` to see",
            "  every mesh's per-chip footprint vs the box's available L1+DRAM.",
            "Try one of: a larger box, a larger mesh (higher TP), or a smaller",
            "  dtype (bfp8_b / bfp4_b weights). The recommender's `verdict` field",
            "  flags TIGHT/NO_FIT pre-flight; if you bypassed that, the OOM is",
            "  the runtime confirmation.",
        ],
    ),
    _Pattern(
        name="Device init / PCIe / cluster open failure",
        regex=re.compile(
            r"(Failed to open device"
            r"|Failed to initialize cluster"
            r"|PCIe device .* not found"
            r"|tt_SiliconDevice.*throw"
            r"|tt_ClusterDescriptor)"
        ),
        suggestion=[
            "The pytest could not even open the requested devices.",
            "Check that the MESH_DEVICE env var matches a label the demo's",
            "  parametrize knows about (see `tt_hw_planner prepare` output —",
            "  it refuses to emit a run if there is no direct MESH_DEVICE label).",
            "Run `tt-smi` to confirm the cards are visible and not held by another",
            "  process. If a previous crashed run is still holding the device,",
            "  `pkill -9 -f pytest` and retry.",
        ],
    ),
    _Pattern(
        name="Mesh shape mismatch vs MESH_DEVICE env",
        regex=re.compile(r"(mesh.*shape.*mismatch" r"|requested mesh .* not available" r"|MeshShape.*!=)"),
        suggestion=[
            "The demo's MESH_DEVICE label resolves to a different physical mesh",
            "  than the one `prepare` requested. Re-run `tt_hw_planner prepare",
            "  <MODEL> --box <BOX> --mesh <R,C>`; it now refuses silent",
            "  substitution and prints the supported labels for the demo.",
        ],
    ),
    _Pattern(
        name="TT_FATAL kernel preconditions (generic)",
        regex=re.compile(r"TT_FATAL.*\("),
        suggestion=[
            "A ttnn kernel asserted at graph-construction. The specific predicate",
            "  is in the matched line. Two paths to fix:",
            "  1. If the predicate corresponds to a static shape/dtype constraint",
            "     (divisibility, tile alignment, layout), add it to",
            "     scripts/tt_hw_planner/kernel_constraints.py so `compat` flags it",
            "     pre-flight for every model that would hit it.",
            "  2. If it's a tuning-table miss, run `tt_hw_planner scaffold <MODEL>",
            "     --apply` to insert defaults.",
        ],
    ),
    _Pattern(
        name="HF Hub auth / 401 / 403",
        regex=re.compile(
            r"(401 Client Error"
            r"|403 Client Error"
            r"|HfHubHTTPError"
            r"|GatedRepoError"
            r"|access to model .* is restricted)"
        ),
        suggestion=[
            "Hugging Face refused weight download (gated repo or missing token).",
            "Either `huggingface-cli login`, or set HF_TOKEN in the environment",
            "  before re-running. This is model-agnostic; affects any gated repo",
            "  (Llama, Gemma, Mistral-Instruct, etc.).",
        ],
    ),
]


_GENERIC_SUGGESTION = [
    "This assertion is not yet matched by any pattern in _postmortem.py.",
    "If you want the planner to catch this pre-flight in future, add a check:",
    "",
    "  1. Open scripts/tt_hw_planner/kernel_constraints.py.",
    "  2. Pick the right check_* function (attention / mlp / rope / rmsnorm).",
    "  3. Add a KernelFinding with the predicate that this assert tests.",
    "  4. Add a matching _Pattern entry to scripts/tt_hw_planner/perf/_postmortem.py",
    "     so the post-mortem can identify it next time.",
    "",
    "The pattern follows the existing entries in kernel_constraints.py:",
    "    out.append(KernelFinding(",
    "        op='<which ttnn op asserted>',",
    "        field='<which config field>', value=<the value>,",
    "        constraint='<plain-english description>',",
    "        passes=<the boolean check that this assert tests>,",
    "        severity=Severity.BLOCKER,",
    "        fix='<what the user can do>',",
    "        source='<file:line where this assert lives>',",
    "    ))",
]


_ASSERTION_RX = re.compile(r"^E?\s*(AssertionError|RuntimeError|ValueError|TypeError):\s*(.*)$")
_FILE_LINE_RX = re.compile(r"^E?\s*File \"([^\"]+)\", line (\d+)")
_FAIL_BANNER_RX = re.compile(r"^=+\s*FAILURES\s*=+\s*$")

_NO_COLLECTION_RX = re.compile(r"collected 0 items")
_PYTEST_PATH_RX = re.compile(r"ERROR: file or directory not found: (\S+)")
_FAULTHANDLER_TIMEOUT_RX = re.compile(r"^\+{20,} Timeout \+{20,}$")

# Fabric-sync-timeout analyzer regexes (used by _parse_fabric_sync_timeout)
_FABRIC_TIMEOUT_LINE_RX = re.compile(r"Fabric Router Sync:\s*Timeout after\s+(\d+)\s*ms\s+on Device\s+(\d+)")
_FABRIC_INIT_OK_RX = re.compile(r"Fabric initialized on (\d+) devices")
_FABRIC_INIT_DEV_OK_RX = re.compile(r"Fabric initialized on Device\s+(\d+)")
_FABRIC_CHAN_STATUS_RX = re.compile(
    r"chan=\s*(\d+)\s+logical=(\S+)\s+\(LOGICAL\)\s+status=(0x[0-9a-fA-F]+)\s+\(([A-Z_]+)\)"
)
_FABRIC_EARLIEST_STAGE_RX = re.compile(r"Earliest init stage reached:\s*([A-Z_]+)\s*\((\d+)\s*core")
_FABRIC_HINT_RX = re.compile(r"Hint:\s+(router\(s\) likely entered the main function[^\n]+)")


def _parse_fabric_sync_timeout(text: str) -> Optional[Dict[str, Any]]:
    """Extract structured evidence from a fabric_firmware_initializer timeout.

    Returns None if the log shows no fabric router sync timeout.

    The evidence dict tells the user exactly which device stalled, which
    channels reached which init stage, and how far the per-device init
    actually got before the timeout. That's the data tt_metal triage will
    ask for — surfacing it pre-emptively saves a round trip.
    """
    timeout_match = _FABRIC_TIMEOUT_LINE_RX.search(text)
    if timeout_match is None:
        return None

    timeout_ms = int(timeout_match.group(1))
    stuck_device = int(timeout_match.group(2))

    devices_initialized: List[int] = []
    for m in _FABRIC_INIT_DEV_OK_RX.finditer(text):
        devices_initialized.append(int(m.group(1)))

    initialized_count: Optional[int] = None
    m = _FABRIC_INIT_OK_RX.search(text)
    if m:
        initialized_count = int(m.group(1))

    stuck_channels: List[Dict[str, Any]] = []
    for m in _FABRIC_CHAN_STATUS_RX.finditer(text):
        stuck_channels.append(
            {
                "chan": int(m.group(1)),
                "logical": m.group(2),
                "status": m.group(3),
                "stage": m.group(4),
            }
        )

    earliest_stage: Optional[str] = None
    stuck_core_count: Optional[int] = None
    m = _FABRIC_EARLIEST_STAGE_RX.search(text)
    if m:
        earliest_stage = m.group(1)
        stuck_core_count = int(m.group(2))

    hint: Optional[str] = None
    m = _FABRIC_HINT_RX.search(text)
    if m:
        hint = m.group(1).rstrip(".")

    return {
        "timeout_ms": timeout_ms,
        "stuck_device": stuck_device,
        "devices_that_initialized": sorted(set(devices_initialized)),
        "fabric_initialized_count": initialized_count,
        "stuck_channels": stuck_channels,
        "earliest_init_stage_reached": earliest_stage,
        "stuck_core_count": stuck_core_count,
        "diagnostic_hint": hint,
    }


# Buffer-overflow analyzer regexes (used by _parse_buffer_overflow)
_OVERFLOW_LINE_RX = re.compile(r"Profiler DRAM buffers were full, markers were dropped!")
_OVERFLOW_DEVICE_RX = re.compile(r"device\s+(\d+)")
_OVERFLOW_CORE_RX = re.compile(r"worker core\s+(\d+),\s*(\d+)")
_OVERFLOW_RISC_RX = re.compile(r"Risc\s+([A-Z_0-9]+)")
_OVERFLOW_CAP_RX = re.compile(r"bufferEndIndex\s*=\s*(\d+)")
_MAX_TOKENS_ARG_RX = re.compile(r"--max[_-]generated[_-]tokens['\"]?,?\s+['\"]?(\d+)")


def _parse_buffer_overflow(text: str) -> Optional[Dict[str, Any]]:
    """Extract structured evidence from `Profiler DRAM buffers were full` lines.

    Returns None if the log shows no overflow warnings.
    """
    overflow_lines = [ln for ln in text.splitlines() if _OVERFLOW_LINE_RX.search(ln)]
    if not overflow_lines:
        return None

    devices: set[int] = set()
    riscs: set[str] = set()
    cores: set[Tuple[int, int]] = set()
    cap: Optional[int] = None
    for ln in overflow_lines:
        m = _OVERFLOW_DEVICE_RX.search(ln)
        if m:
            devices.add(int(m.group(1)))
        m = _OVERFLOW_RISC_RX.search(ln)
        if m:
            riscs.add(m.group(1))
        m = _OVERFLOW_CORE_RX.search(ln)
        if m:
            cores.add((int(m.group(1)), int(m.group(2))))
        m = _OVERFLOW_CAP_RX.search(ln)
        if m and cap is None:
            cap = int(m.group(1))

    # Configured token count from the pytest argv echoed at top of log
    configured: Optional[int] = None
    m = _MAX_TOKENS_ARG_RX.search(text)
    if m:
        configured = int(m.group(1))

    # Figure out WHEN the overflow fired relative to the demo's lifecycle.
    # This is the highest-signal feature: if pytest already wrote `PASSED`
    # before the first overflow, the demo SUCCEEDED and the crash is purely
    # tracy's end-of-test post-processing — a completely different kind of
    # failure from "the model itself broke during decode".
    first_overflow_idx: Optional[int] = None
    start_profiler_idx: Optional[int] = None
    decode_metrics_idx: Optional[int] = None
    pytest_passed_idx: Optional[int] = None
    for i, ln in enumerate(text.splitlines()):
        if first_overflow_idx is None and _OVERFLOW_LINE_RX.search(ln):
            first_overflow_idx = i
        if start_profiler_idx is None and "Start profiler" in ln:
            start_profiler_idx = i
        if decode_metrics_idx is None and ("Time to First Token" in ln or "TTFT" in ln or "tok/s" in ln):
            decode_metrics_idx = i
        if pytest_passed_idx is None and ("PASSED" in ln or "passed in" in ln):
            pytest_passed_idx = i
        if (
            first_overflow_idx is not None
            and start_profiler_idx is not None
            and decode_metrics_idx is not None
            and pytest_passed_idx is not None
        ):
            break

    phase = "unknown"
    if first_overflow_idx is not None:
        # Note: pytest's `PASSED` marker can land on the SAME log line as the
        # first Metal warning when stderr/stdout interleave, so we use >= here.
        # That correctly flags "demo passed, tracy crashed at teardown".
        if pytest_passed_idx is not None and first_overflow_idx >= pytest_passed_idx:
            phase = "post_test_teardown_after_PASSED"
        elif decode_metrics_idx is not None and first_overflow_idx > decode_metrics_idx:
            phase = "after_decode_metrics"
        elif start_profiler_idx is not None and first_overflow_idx > start_profiler_idx:
            phase = "during_test"
        else:
            phase = "prefill_or_warmup"

    # Did `--dump-device-data-mid-run` actually do anything? Count real flush
    # events (not the argv echo at the top). If 0, the option is a no-op in
    # this tracy version and halving tokens won't help us.
    mid_run_dump_events = 0
    for ln in text.splitlines():
        # The argv echo line starts with `# argv: [...]`; skip that.
        if ln.startswith("# argv:"):
            continue
        if (
            "MidRun" in ln
            or "mid_run_dump" in ln
            or "periodicDeviceDump" in ln
            or "MID_RUN_DUMP" in ln
            and "=" in ln  # actual flush, not env-var echo
        ):
            mid_run_dump_events += 1

    return {
        "overflow_warnings": len(overflow_lines),
        "buffer_capacity": cap if cap is not None else 12000,
        "devices_affected": sorted(devices),
        "risc_types_affected": sorted(riscs),
        "cores_affected": sorted(cores),
        "configured_max_tokens": configured,
        "overflow_phase": phase,
        "mid_run_dump_events": mid_run_dump_events,
    }


def _recommend_retry_for_overflow(
    ev: Dict[str, Any],
    prev_ev: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Translate buffer-overflow evidence into concrete retry kwargs.

    Returns (retry_kwargs, explanation). `retry_kwargs is None` means the
    framework concluded this failure is NOT auto-recoverable by the
    knobs we control — the caller should stop retrying and surface the
    structured evidence to the user.

    Decision ladder (checked in order; the first hit returns):

      A. Demo PASSED but tracy crashed at end-of-test → tracy bug, not
         model bug. No retry can fix it. Loudly escalate.

      B. The previous retry produced IDENTICAL overflow evidence (same
         count, same cores). Halving tokens didn't help. Escalate.

      C. mid-run dump option is a no-op (no actual flush events in log).
         The single knob we have isn't working. Escalate.

      D. Overflow fired in prefill/warmup. Token count can't help. Escalate.

      E. Otherwise → halve `--max-generated-tokens`.
    """
    # A. Demo PASSED — tracy is the failure, not the model.
    if ev.get("overflow_phase") == "post_test_teardown_after_PASSED":
        return None, (
            "the DEMO itself passed (pytest reported PASSED before tracy "
            "asserted); the crash is purely tracy's end-of-test post-processing. "
            "No `perf collect` flag can fix this — see escalation guidance."
        )

    # B. Identical evidence → knob isn't moving the needle.
    if prev_ev is not None and _evidence_essentially_equal(ev, prev_ev):
        return None, (
            "previous retry produced IDENTICAL overflow evidence "
            f"(overflow_warnings={ev['overflow_warnings']}, "
            f"cores_affected={len(ev['cores_affected'])}); halving tokens has no "
            "effect, so the overflow is NOT decode-length-driven. Escalating."
        )

    # C. mid-run dump isn't actually flushing.
    if ev.get("mid_run_dump_events", 0) == 0:
        return None, (
            "no mid-run flush events appear in the log even though "
            "--dump-device-data-mid-run is on; the tracy option is a no-op in "
            "this build. Halving tokens won't help either. Escalating."
        )

    # D. Overflow fired in prefill — tokens don't matter.
    if ev.get("overflow_phase") in ("prefill_or_warmup", "during_test"):
        return None, (
            "first overflow fired before decode metrics were written; "
            "lowering --max-generated-tokens cannot help. Escalating."
        )

    # E. Legacy: halve token count.
    current = ev.get("configured_max_tokens") or 8
    next_tokens = max(1, current // 2)
    if next_tokens >= current:
        return None, (f"already at --max-generated-tokens={current}; cannot halve further.")
    explanation = (
        f"buffer overflowed at {ev['overflow_warnings']} (device, core, RISC) sites "
        f"with current --max-generated-tokens={current}; halving to {next_tokens}."
    )
    return {"max_generated_tokens": next_tokens}, explanation


def _evidence_essentially_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Compare two buffer-overflow evidence dicts to detect a no-op retry.

    We say two pieces of evidence are "essentially equal" if the overflow
    profile (count, cores, devices, RISCs) is the same, even if other fields
    (configured_max_tokens, timings) differ. Used by the retry loop to detect
    when halving tokens did nothing.
    """
    return (
        a.get("overflow_warnings") == b.get("overflow_warnings")
        and len(a.get("cores_affected", [])) == len(b.get("cores_affected", []))
        and a.get("devices_affected") == b.get("devices_affected")
        and a.get("risc_types_affected") == b.get("risc_types_affected")
    )


def _escalation_guidance(ev: Optional[Dict[str, Any]] = None) -> List[str]:
    """Human-actionable next steps when the auto-retry decides it cannot recover.

    The guidance is RANKED by what we know about the failure. The cheapest
    fix is first. If `ev` shows the demo PASSED (i.e. tracy is the only
    failure), we lead with the tracy-only fixes since they're the smallest
    surface change.
    """
    demo_passed = ev is not None and ev.get("overflow_phase") == "post_test_teardown_after_PASSED"
    out: List[str] = []
    out.append("Auto-retry halted. Next steps, cheapest first:")
    out.append("")

    if demo_passed:
        out += [
            "  IMPORTANT: the demo itself PASSED. The model works at this",
            "  (mesh, dtype, --max-generated-tokens). Only tracy's end-of-test",
            "  post-processing failed. That means the cheapest fix is to make",
            "  tracy tolerant of dropped markers — not to change the model run.",
            "",
            "  a) [smallest] Patch tracy to warn-and-skip on missing device data.",
            "     Edit tools/tracy/process_ops_logs.py around line 516:",
            "         assert candidates, ...",
            "     becomes:",
            "         if not candidates:",
            "             missing_ops_count += 1",
            "             continue",
            "     and log the total at end of process_ops. You get a partial",
            "     ops_perf_results.csv (missing the device 3 ops that overflowed)",
            "     which is still enough for the baseline + roofline. Optimizer",
            "     blocks that depend on those specific ops will mark themselves",
            "     UNKNOWN instead of GREEN/RED — that's the correct degradation.",
            "",
            "  b) Validate the framework end-to-end on a model whose marker",
            "     count stays under 12000/RISC. A 1.5B-class model on QB2",
            "     mesh=[1,4] fits comfortably. Use that as the first baseline",
            "     while the tracy fix lands.",
            "",
            "  c) [most invasive] File against tools/tracy for a real fix:",
            "     either grow the per-RISC buffer (currently 12000) or make",
            "     `--dump-device-data-mid-run` actually flush during the run.",
            "     run_meta.json -> postmortem.quantified has the exact",
            "     (device, core, RISC) tuples that overflowed.",
        ]
    else:
        out += [
            "  a) Use a smaller model OR a smaller mesh. Markers/RISC scales with",
            "     (num_layers x ops_per_layer x cores_per_op). Try a 1.5B-class",
            "     model first to validate the framework end-to-end.",
            "",
            "  b) Patch tracy locally to be tolerant of missing device data.",
            "     Edit tools/tracy/process_ops_logs.py line 516:",
            "         assert candidates, ... -> if not candidates: continue",
            "     Gives a partial-but-usable CSV.",
            "",
            "  c) File against tools/tracy for a real fix: grow the per-RISC",
            "     buffer or fix `--dump-device-data-mid-run` to actually flush.",
        ]
    return out


def analyze_log(
    log_path: Path,
    max_tail_lines: int = 4000,
    prev_evidence: Optional[Dict[str, Any]] = None,
    current_mesh: Optional[Tuple[int, int]] = None,
) -> PostMortem:
    """Read `collect.log` and produce a structured post-mortem.

    Robust against:
    - Tracy's noisy preamble (we scan from the FAILURES banner backwards if
      one is present, else fall back to the last N lines).
    - Truncated logs.
    - Logs with no failure (returns an empty PostMortem).
    """
    if not log_path.exists():
        return PostMortem(matched_pattern=None, assertion_text=None, file_line=None)

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return PostMortem(matched_pattern=None, assertion_text=None, file_line=None)

    lines = text.splitlines()
    if len(lines) > max_tail_lines:
        # Scan only the tail; that's where pytest dumps its failure report.
        lines = lines[-max_tail_lines:]

    # ---- Class 1: pytest never collected any test (pre-traceback failure) ----
    # Catches: argv-mangling bugs, missing test files, bad -k selectors.
    no_collection = any(_NO_COLLECTION_RX.search(ln) for ln in lines)
    path_errors = [m.group(1) for ln in lines for m in [_PYTEST_PATH_RX.search(ln)] if m]
    if no_collection and path_errors:
        return _argv_mangling_postmortem(path_errors, lines)

    # ---- Class 2: faulthandler timeout banner without a real assertion ------
    # The HF download can sit inside `xet_get` long enough that pytest's
    # `--timeout` thread dumps stacks. If the run still completed, we don't
    # surface this. If it didn't complete *and* there's no other signal, the
    # download itself is the prime suspect.
    timeout_seen = any(_FAULTHANDLER_TIMEOUT_RX.search(ln) for ln in lines)

    assertion_text: Optional[str] = None
    file_line: Optional[Tuple[str, int]] = None
    last_file_line: Optional[Tuple[str, int]] = None

    for ln in lines:
        m = _FILE_LINE_RX.search(ln)
        if m:
            last_file_line = (m.group(1), int(m.group(2)))
            continue
        m = _ASSERTION_RX.search(ln)
        if m and assertion_text is None:
            assertion_text = m.group(2).strip()
            file_line = last_file_line
            # Don't break: keep last assertion (the most recent / innermost).

    if assertion_text is None:
        # Try one more thing: NOC_MAX_BURST_SIZE static_assert from llkbuild
        for ln in lines:
            if "static_assert" in ln and "NOC_MAX_BURST_SIZE" in ln:
                assertion_text = ln.strip()
                break

    matched = _match_known_pattern(assertion_text)
    # Some failure classes have a Metal-side warning that fires BEFORE pytest's
    # AssertionError, or a multi-line TT_THROW whose `info:` continuation is on
    # a separate line. Sniff the full log for those so we catch the class even
    # when the parsed assert text alone wouldn't match.
    #
    # IMPORTANT: check the fabric pattern before the generic `TT_FATAL` match
    # would (in _match_known_pattern) — the fabric one carries actionable
    # tt-smi guidance and structured evidence, the generic TT_FATAL only
    # suggests adding a kernel constraint, which is wrong for environmental
    # init failures.
    if _FABRIC_TIMEOUT_LINE_RX.search(text):
        matched = _FABRIC_SYNC_TIMEOUT_PATTERN
    elif matched is None and _OVERFLOW_LINE_RX.search(text):
        matched = _BUFFER_OVERFLOW_PATTERN

    suggestion: List[str]
    matched_name: Optional[str]
    if matched is not None:
        matched_name = matched.name
        suggestion = list(matched.suggestion)
    elif timeout_seen and assertion_text is None:
        matched_name = "HF/network stall during weight download"
        suggestion = list(_HF_DOWNLOAD_STALL_SUGGESTION)
    else:
        matched_name = None
        suggestion = list(_GENERIC_SUGGESTION)

    quantified: Dict[str, Any] = {}
    next_retry_args: Optional[Dict[str, Any]] = None
    retry_explanation: Optional[str] = None

    # Fabric router sync timeout: extract structured evidence AND propose a
    # one-shot bisect to mesh=(1, 1). Single-chip skips fabric init entirely,
    # so the bisect outcome is dispositive (model vs link). No reset is
    # attempted from inside the framework — that would be too invasive on a
    # shared box.
    if matched is _FABRIC_SYNC_TIMEOUT_PATTERN:
        ev = _parse_fabric_sync_timeout(text)
        if ev is not None:
            quantified = ev
            retry_kwargs, retry_explanation = _recommend_retry_for_fabric_timeout(
                ev, prev_evidence, current_mesh=current_mesh
            )
            if retry_kwargs is not None:
                next_retry_args = retry_kwargs
            else:
                suggestion = (
                    list(suggestion)
                    + [""]
                    + [
                        "Bisect is exhausted. Surface the postmortem.txt + the log",
                        "  to tt_metal — the structured evidence (stuck device, init",
                        "  stage) is exactly what triage will need.",
                    ]
                )

    # The buffer-overflow pattern carries a real analyzer. Any future pattern
    # that wants quantified evidence + auto-retry should add a sibling here.
    if matched is _BUFFER_OVERFLOW_PATTERN:
        ev = _parse_buffer_overflow(text)
        if ev is not None:
            quantified = ev
            retry_kwargs, retry_explanation = _recommend_retry_for_overflow(ev, prev_evidence)
            if retry_kwargs is not None:
                next_retry_args = retry_kwargs
            else:
                # Auto-retry can't help. Append escalation guidance to the
                # suggestion so render() prints it alongside the evidence.
                suggestion = list(suggestion) + [""] + _escalation_guidance(ev)

    pm = PostMortem(
        matched_pattern=matched_name,
        assertion_text=assertion_text,
        file_line=file_line,
        traceback_tail=[ln for ln in lines[-30:] if ln.strip()],
        suggested_action=suggestion,
        quantified=quantified,
        next_retry_args=next_retry_args,
        retry_explanation=retry_explanation,
    )
    return pm


def _argv_mangling_postmortem(path_errors: List[str], lines: List[str]) -> PostMortem:
    """Failure mode: pytest received argv it could not parse.

    The canonical instance is `tools/tracy/__main__.py` rejoining argv with
    plain `" ".join(...)` and feeding it to `shell=True`, which re-splits
    `-k 'performance and batch-1'` into four words. pytest then thinks
    `and` is a path.

    Distinct from a Python assertion, so we return a hand-crafted
    PostMortem rather than going through `_match_known_pattern`.
    """
    return PostMortem(
        matched_pattern="pytest argv mangled (likely shell-quoting bug in tracy wrapper)",
        assertion_text=(f"pytest collected 0 items; " f"interpreted {path_errors[:3]} as test file paths."),
        file_line=None,
        traceback_tail=[ln for ln in lines[-20:] if ln.strip()],
        suggested_action=[
            "pytest never collected any test in this run. The most common cause",
            "is that an argv item containing spaces (e.g. `-k 'A and B'`) was",
            "re-tokenized by a shell layer between us and pytest.",
            "",
            "Check `_build_tracy_argv` in scripts/tt_hw_planner/perf/collect.py:",
            "every argv item handed to `python -m tracy` MUST be passed through",
            "`_shell_safe` / `shlex.quote`, because tracy joins argv with plain",
            '`" ".join(...)` and runs it under `shell=True`.',
            "",
            "If your fork has the fix, then re-check that `inv.args` from",
            "bringup.py doesn't include a bare path-like word that overlaps a",
            "real on-disk file (e.g. `and` next to a directory called `and/`).",
        ],
    )


_HF_DOWNLOAD_STALL_SUGGESTION = [
    "pytest's faulthandler dumped thread stacks (the `+++ Timeout +++` banner)",
    "without a Python AssertionError. The most common cause is an HF Hub",
    "download taking longer than `pytest.ini`'s `timeout` setting.",
    "",
    "Fixes (cheapest first):",
    "  1. Pre-download the model into the local HF cache before `perf collect`:",
    "         huggingface-cli download <MODEL>",
    "     then re-run `tt_hw_planner perf collect`.",
    "  2. Set `HF_HUB_DOWNLOAD_TIMEOUT=600` in the environment when collecting.",
    "  3. Raise the pytest timeout for that one run with",
    "         pytest ... -o timeout=1800",
    "     by extending bringup.py's invocation builder.",
]


def _match_known_pattern(assertion_text: Optional[str]) -> Optional[_Pattern]:
    if not assertion_text:
        return None
    for p in _KNOWN_PATTERNS:
        if p.regex.search(assertion_text):
            return p
    return None
