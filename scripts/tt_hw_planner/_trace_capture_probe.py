from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_LADDER = [
    (
        "residency",
        r"per-layer weight|load\+evict|load/evict|lazy per-layer|does not fit|stream[^\n]{0,24}(weight|layer|expert)"
        r"|evict[^\n]{0,24}(weight|layer|expert)|(weight|layer|expert)[^\n]{0,24}evict",
        "Weights stream from host per layer (build/evict). Make them RESIDENT: shard the parameter-heavy "
        "layers across the mesh so each chip holds its slice for the whole run (for an MoE that is "
        "expert-parallel via get_shard_plan under the host-free goal; for a dense model, higher TP or "
        "pipeline-parallel), and build every layer ONCE at init instead of build-then-evict. The runtime "
        "op-signature probe is the authoritative generic check: many distinct host->device weight uploads "
        "during a forward == streaming.",
    ),
    (
        "token_feed",
        r"torch\.cat\s*\(|from_torch\([^)]*ids|\.item\(\)",
        "Decode feeds the next token via host (torch.cat / re-upload / .item()). Feed it ON DEVICE: write "
        "the ttnn.argmax result into a fixed [1,1] input tensor on device and advance an on-device position "
        "index; do not rebuild the input on host.",
    ),
    (
        "kv_cache",
        None,
        "No persistent KV cache -> full re-prefill per token. Add an on-device KV cache (persistent K/V, "
        "single-token append at the current position); see models/tt_transformers/tt/attention.py.",
    ),
    (
        "decode_step",
        None,
        "No fixed-shape, trace-capturable decode_step(state). Add a single-token decode_step whose tensor "
        "shapes are constant every step, then expose trace_capture_selftest(device) that wraps it in "
        "ttnn.begin_trace_capture / end_trace_capture.",
    ),
]


def _read(demo_dir: Path) -> str:
    text = []
    for sub in ("tt", "_stubs"):
        d = demo_dir / sub
        if d.is_dir():
            for p in sorted(d.glob("*.py")):
                try:
                    text.append(p.read_text())
                except OSError:
                    pass
    return "\n".join(text)


def probe(demo_dir: Path) -> dict:
    src = _read(demo_dir)
    blockers = []

    for name, pat, guidance in _LADDER:
        if pat is not None:
            if re.search(pat, src):
                blockers.append({"rung": name, "guidance": guidance})
        elif name == "kv_cache":
            has_kv = re.search(r"kv[_ ]?cache|update_cache|paged_update|cache_k\b|past_key|kvcache", src, re.I)
            reprefill = re.search(r"for .* in range\(.*max_new|re-?prefill|full recompute", src, re.I)
            if reprefill and not has_kv:
                blockers.append({"rung": name, "guidance": guidance})
        elif name == "decode_step":
            if not re.search(r"def decode_step|def decode_trace|begin_trace_capture", src):
                blockers.append({"rung": name, "guidance": guidance})

    trace_ready = not blockers
    device_capture = None
    if trace_ready:
        device_capture = _device_selftest(demo_dir)

    return {
        "trace_ready": bool(trace_ready and (device_capture is None or device_capture.get("ok"))),
        "static_blockers": blockers,
        "device_capture": device_capture,
    }


def _device_selftest(demo_dir: Path) -> dict | None:
    try:
        sys.path.insert(0, str(demo_dir.parent))
        mod = __import__("%s.tt.pipeline" % demo_dir.name, fromlist=["pipeline"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "cannot import tt.pipeline: %s" % e}
    fn = getattr(mod, "trace_capture_selftest", None)
    if fn is None:
        return None
    try:
        ok = bool(fn())
        return {"ok": ok, "reason": "" if ok else "trace_capture_selftest returned False"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "begin_trace_capture raised (host op in decode step): %s" % e}


def main(argv):
    demo_dir = Path(argv[1]).resolve()
    r = probe(demo_dir)
    print("TRACE_PROBE=" + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
