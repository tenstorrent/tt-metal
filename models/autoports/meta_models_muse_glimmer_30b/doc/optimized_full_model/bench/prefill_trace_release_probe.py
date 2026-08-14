# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which step of a prefill-trace *release* trips the fabric ERISC watcher assert.

Running ``test_prefill_trace_survives_rebinding_the_same_external_cache`` under watcher
stops the device with

    Device 0 acteth core(x=0,y=9) virtual(x=29,y=25): subordinate_erisc detected invalid
    NOC command buffer state before starting the next kernel (write-capable NOC packet
    tags must be zero so implicit transaction ID users start with transaction ID 0).
    Current kernel: fabric_erisc_router.cpp

while ``test_prefill_trace_is_opt_in_and_matches_the_eager_path`` -- which captures a
prefill trace and replays it, but never releases one -- is watcher-clean.  A prefill
trace contains fabric collectives, so the suspect is releasing one.  This bisects the
release into its three parts, each arm on a fresh process:

``capture``        capture a prefill trace and replay it. The clean control.
``release``        capture, then release the trace and free its buffers.
``recapture``      capture, release, capture again, replay again.
``clone_cache``    clone the 104 KV-cache tensors and free them, no trace release at all.

Run each arm under watcher, one at a time, resetting the devices in between::

    TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1 \\
    TT_METAL_LOGS_PATH=<dir> python .../prefill_trace_release_probe.py --arm release
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
    clear_generator_cache,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

OUT = ROOT / "doc/optimized_full_model"
ARMS = ("capture", "release", "recapture", "clone_cache")


def say(*args) -> None:
    print(*args, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--length", type=int, default=96)
    args = parser.parse_args()

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    try:
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            layer_indices=[int(i) for i in args.layers.split(",")],
            reuse=False,
            prefill_trace=True,
        )
        torch.manual_seed(3)
        prompt = [int(t) for t in torch.randint(0, generator.model.config.vocab_size, (args.length,)).tolist()]
        own = generator.model.kv_cache

        if args.arm == "clone_cache":
            clones = [
                [ttnn.clone(k, memory_config=k.memory_config()), ttnn.clone(v, memory_config=v.memory_config())]
                for k, v in own
            ]
            ttnn.synchronize_device(mesh)
            say(f"PR cloned {sum(len(p) for p in clones)} cache tensors")
            for pair in clones:
                for tensor in pair:
                    ttnn.deallocate(tensor)
            ttnn.synchronize_device(mesh)
            say("PR freed the clones")
            say("PR_OK arm=clone_cache")
            return 0

        generator.prefill_forward(torch.tensor([prompt]), kv_cache=own, prompt_lens=[len(prompt)])
        assert generator._prefill_traces, "the arm needs a captured prefill trace"
        say(f"PR captured {sorted(generator._prefill_traces)}")
        if args.arm == "capture":
            generator.reset()
            generator.prefill_forward(torch.tensor([prompt]), kv_cache=own, prompt_lens=[len(prompt)])
            say("PR replayed")
            say("PR_OK arm=capture")
            return 0

        generator._release_prefill_traces()
        say("PR released")
        if args.arm == "release":
            say("PR_OK arm=release")
            return 0

        generator.reset()
        generator.prefill_forward(torch.tensor([prompt]), kv_cache=own, prompt_lens=[len(prompt)])
        say(f"PR recaptured {sorted(generator._prefill_traces)} and replayed")
        say("PR_OK arm=recapture")
        return 0
    finally:
        if generator is not None:
            generator.teardown()
        clear_generator_cache()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
