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
``rebuild``        capture, tear the generator down (which releases the trace), then build a
                   *second* generator on the same still-open mesh and prefill+decode with it.

The first four arms are all negative controls: none of them builds a second model after a
release, so a clean result from them bounds nothing.  ``rebuild`` was added as the
positive-arm *candidate* -- it is the exact sequence this stage once blamed for the fabric
ERISC assert ("release a prefill trace, then build and run another model on the same
mesh"), which is also what running the two opt-in tests in one pytest process does through
the module fixture.

It comes back **clean** (``logs/watcher_probe_rebuild.log``, ``watcher_probe_rebuild/``:
``WATCHER_CLEAN``, 4 attach / 4 detach), and so does the two-test pair.  That does not
retract the assert -- it reproduces at teardown when those two tests share a process with
the other ten gated cases -- but it does retract the *statement* of it: that sequence is
not sufficient.  See README limitation 6.  This arm is kept because it is the cheapest way
for a later stage to re-test the sequence directly.

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
ARMS = ("capture", "release", "recapture", "clone_cache", "rebuild")


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
    built: list = []

    def build(**kwargs):
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            layer_indices=[int(i) for i in args.layers.split(",")],
            reuse=False,
            **kwargs,
        )
        built.append(generator)
        return generator

    try:
        generator = build(prefill_trace=True)
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

        if args.arm == "rebuild":
            # The named sequence, in one process on the still-open mesh: tear the first
            # generator down (``teardown()`` calls ``_release_prefill_traces()``), then
            # build a second model and actually run it.  This is what the module fixture
            # does between the two opt-in tests.
            generator.teardown()
            built.remove(generator)
            clear_generator_cache()
            say("PR tore down the first generator (its prefill trace was released)")
            second = build()
            say("PR built a second generator on the same mesh")
            second.prefill_forward(
                tokens=torch.tensor([prompt], dtype=torch.long),
                kv_cache=second.model.kv_cache,
                prompt_lens=[len(prompt)],
            )
            token = second.decode_forward(
                tokens=torch.tensor([[prompt[-1]]], dtype=torch.long),
                start_pos=torch.tensor([len(prompt)], dtype=torch.int32),
                kv_cache=second.model.kv_cache,
                sample_on_device=True,
            )
            say(f"PR ran prefill+traced-decode on the second generator (token={int(token[0])})")
            say("PR_OK arm=rebuild")
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
        for generator in built:
            generator.teardown()
        clear_generator_cache()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
