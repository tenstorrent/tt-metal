# Stage 06 blocker: re-diagnosed

Operator investigation after stage 06 marked itself blocked with "an operator
must reboot the host before further TT hardware work". **No host reboot is
needed. The hardware is healthy.** The real blocker is a hang in the decode
all-reduce, and it is reproducible.

## What was actually measured

**1. The devices are fine.** With no process holding `/dev/tenstorrent`, a fresh
`MeshShape(1,4)` open + `FABRIC_1D_RING` init + `create_global_semaphore` +
close passes cleanly, repeatedly, and all four p300c enumerate. Two independent
`tt-smi -r` cycles each restored a working mesh.

**2. The "model-setup mesh-buffer write stall" is not intrinsic.** The stack at
the time of the apparent stall was:

    from_torch (ttnn/operations/core.py:352)
      _shard (multichip_decoder.py:79)
      _shard_decode_weight (multichip_decoder.py:91)
      from_state_dict (multichip_decoder.py:193)   <- weights["mlp_gate_decode"]

That call converts one tensor: `mlp.gate_proj.weight`, `[5120, 17408]`, sharded
on dim -1 across four ranks at BFP4. Timed in isolation on healthy devices:

| conversion | time |
|---|---:|
| `[5120,17408]` BFP4, ShardTensorToMesh dim -1 (the "stuck" one) | **0.88 s** |
| `[5120,17408]` BF16 control | 0.39 s |
| `[5120,1024]` BFP4 control | 0.05 s |

So a call that takes 0.88 s was observed spinning at 100 % CPU for ~50 minutes.
It is not slow; it hangs, and only on devices left in a degraded state.
**After a board reset, `from_state_dict` completes in about 2 minutes.**

**3. The real hang is in the collective.** On freshly reset devices the repro
gets past setup and then stops here, at 100 % CPU, unchanged across samples at
+1, +2 and +3 minutes:

    _all_reduce (multichip_decoder.py:441)
      _tp_linear (multichip_decoder.py:531)
      _full_attention_decode (multichip_decoder.py:916)
      _token_mixer_decode (functional_decoder.py:384)
      decode_forward (optimized_decoder.py:1696)
      main (tests/full_attention_inactive_kv.py:53)

A decode all-reduce is microseconds. This is a hang on the **first decode step**
of `full_attention_inactive_kv.py`: one full-attention layer (`layer_idx=3`),
`batch=2`, `max_context=64`, `page_size=64`, with an **inactive row**
(`active = [1, 0]`).

Stages 04 and 05 ran this same ring all-reduce successfully at batch 1 and batch
32, so the failing ingredient is the batch-2 / inactive-row configuration, not
the collective in general.

## The feedback loop that produced the wrong conclusion

Killing a process while it is inside a device operation leaves the devices in a
state where the **next** run hangs *earlier* — in `from_torch` during setup
rather than in the collective. That looks like hardware degrading run over run,
which is what motivated the reboot request. It is cleared by `tt-smi -r`.

Sequence to avoid: hang -> kill -> next run hangs in setup -> conclude the
hardware is failing. **Always `tt-smi -r` and re-verify with a mesh smoke after
killing a process mid-device-op**, before drawing any conclusion from the next
run.

## What to work on

The bug to fix is the decode all-reduce hang at `multichip_decoder.py:441` for
batch 2 with an inactive row — not device recovery. Suggested first cuts:

- Does it hang with `active = [1, 1]` at batch 2? That separates "batch 2" from
  "inactive row" as the trigger.
- Does it hang at batch 2 with `max_context`/`page_size` larger than 64?
- `_all_reduce`'s inputs for an inactive row: is a zero/degenerate shard being
  fed to the ring, or a semaphore not being signalled on the inactive rank?

## Separately: the Watcher run cannot start at all

`logs/full_model_watcher_reduced_final.log` fails before model construction:
Watcher instrumentation makes ACTIVE_ETH fabric firmware 27,920 bytes against a
25,600-byte hardware kernel-config buffer limit. That is a real tooling limit,
independent of the hang above and not fixable by a reset. Stage 04 worked around
the Blackhole eth-watcher problem with `TT_METAL_WATCHER_DISABLE_ETH=1`, which
is not mentioned in the stage-06 triage and is the obvious thing to try before
declaring the Watcher evidence unobtainable.
