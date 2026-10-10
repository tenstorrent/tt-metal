# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-metal's prefill_runner.main for GLM-5.3-Flash on the LoudBox, unchanged, with one hook (as the Xing serve stack's
runner, models/demos/xing40_a4b_d_p/serve/runner.py): the runtime's hidden_sink. After every chunk (its layer acks
already sent) the last layer's output goes through the final norm on the device; the row of the chunk's last real
token comes to the host and through the LM head (fp32, CPU). The logits land in GLM_SERVE_LOGITS_DIR as
s<slot>_e<end>_<seq>.bin: one JSON header line (seq, slot, start, end, argmax, finite) then vocab float32 values,
written to a temp name and renamed. A feeder (tests/test_decode_via_runner.py) turns them into tokens: decode via
prefill, every token a prefill of the chunk holding it.

    python -m models.demos.glm53_flash_d_p_lb.serve.runner
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(f"[glm serve runner] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def logits_sink(rt, out_dir: Path):
    import torch

    import ttnn
    from models.demos.glm53_flash_d_p.reference.weights import WeightLoader
    from models.demos.glm53_flash_d_p.tt.common import replicated_to_host, split_to_host
    from models.demos.glm53_flash_d_p.tt.runners.adapter import resolve_model_path

    t0 = time.time()
    lm_head = WeightLoader(resolve_model_path()).get("lm_head.weight").float()
    log(f"LM head {tuple(lm_head.shape)} fp32 on the host in {time.time() - t0:.0f} s")
    out_dir.mkdir(parents=True, exist_ok=True)
    st = {"seq": 0}

    def sink(h, slot: int, start: int, end: int) -> None:
        t = time.time()
        hidden = rt.model.final_norm(h)
        host = (split_to_host(hidden) if rt.model.layout == "split" else replicated_to_host(hidden)).float()
        ttnn.deallocate(hidden)
        logits = torch.nn.functional.linear(host.reshape(-1, host.shape[-1])[end - 1 - start], lm_head).contiguous()
        st["seq"] += 1
        head = {
            "seq": st["seq"],
            "slot": slot,
            "start": start,
            "end": end,
            "argmax": int(logits.argmax()),
            "finite": bool(torch.isfinite(logits).all()),
        }
        name = f"s{slot}_e{end}_{st['seq']}.bin"
        tmp = out_dir / f".{name}.tmp"
        with open(tmp, "wb") as f:
            f.write((json.dumps(head) + "\n").encode())
            f.write(logits.numpy().tobytes())
        os.replace(tmp, out_dir / name)
        log(f"chunk slot {slot} [{start}, {end}): next {head['argmax']} (logits {time.time() - t:.2f} s)")

    return sink


def main() -> int:
    from models.demos.common.prefill.runners import prefill_runner as PR

    logits_dir = Path(os.environ["GLM_SERVE_LOGITS_DIR"])
    build = PR.ADAPTER.build_runtime

    def serving_build(**kw):
        rt = build(**kw)
        rt.hidden_sink = logits_sink(rt, logits_dir)
        log("runtime built; RUNNER_READY")
        return rt

    PR.ADAPTER.build_runtime = serving_build
    try:
        PR.main()
    except BaseException as err:
        log(f"runner: {type(err).__name__}: {err}\n{traceback.format_exc()}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
