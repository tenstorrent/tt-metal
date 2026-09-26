# Sliding-attention receive-buffer race

## Cause

Gemma4's sliding-attention layers shared one pair of persistent K/V receive
buffers. These buffers hold the **halo**: neighboring tokens needed by a device
whose sliding-attention window crosses a context-parallel (CP) shard boundary.

CP devices can progress through layers at different speeds. A faster device can
send the next layer's halo into its neighbor's receive buffer while that neighbor
is still reading the current layer's halo. Local operation ordering does not
ensure that the remote reader has finished. Synchronizing the mesh after each
chunk cannot prevent this overlap between layers within the chunk.

This changes attention outputs and propagates through later layers. The
investigation reproduced differences in raw device KV tensors before the CPU
PCC calculation. The first observed full-model KV mismatch was at layer 13,
confined to one 1,024-token CP shard, consistent with where the reported runs
first diverged.

## Model-side fix

Each sliding-attention layer now owns a separate K/V receive-buffer pair, selected
by `gather_buffer_key=self.layer_idx`. Buffers remain reusable across chunks,
which finish with a mesh synchronization.

The change adds approximately **104 MiB per device**. It does not change the
attention arithmetic, KV-cache layout, or accuracy thresholds. It isolates the
buffers in Gemma4; the standalone TTNN reproducer can still exercise shared-buffer
reuse directly.

Relevant code:

- [Attention call](tt/attention/__init__.py): supplies the layer identity.
- [Ring attention](tt/attention/ring_prefill.py): includes that identity in the receive-buffer keys.
- [CCL manager](tt/ccl.py): allocates and retains the receive buffers.

## Standalone reproducer

[repro_sliding_attention.py](tests/repro_sliding_attention.py) calls
`ttnn.transformer.ring_joint_scaled_dot_product_attention` directly on an 8x4
Galaxy. It uses deterministic sinusoidal tensors and requires no model weights,
tokenizer, or GPU capture.

The script computes two reference outputs with a mesh synchronization after each
attention call. It then captures 128 alternating calls in a trace and compares
every output with its corresponding reference using exact elementwise equality.
The default run replays that trace 200 times.

Run from the repository root with a working TT-Metal build:

```bash
source python_env/bin/activate
export PYTHONPATH="$PWD" TT_METAL_HOME="$PWD"

python models/demos/gemma4_d_p/tests/repro_sliding_attention.py
python models/demos/gemma4_d_p/tests/repro_sliding_attention.py --separate-buffers
```

The default case shares one receive-buffer pair across all calls. The control
uses a separate pair per captured call. The shared-buffer failure is
timing-dependent; increase `--repeats` if it does not appear in a particular run.
A passing finite run does not prove the race is absent.

Observed shared-buffer failure:

```text
Mismatch: replay=61 call=63 device=(1, 3) changed=32669 first=[0, 0, 0, 0] max_abs=0.0022125244140625
```

`device` is the `(CP row, TP column)` coordinate. `first` indexes the local output
as `[batch, query head, query token, channel]`. The mismatch raises an assertion;
the separate-buffer control completed all 25,600 calls with exact matches.

## Verification

| Check | Result |
| --- | --- |
| Standalone TTNN, shared receive buffers | Reproduced 32,669 changed output values |
| Standalone TTNN, separate receive buffers | 25,600 calls matched their synchronized references exactly |
| Full model with separate buffers, one allocated slot | 30 executions of 256K produced bit-identical KV tensors across all 60 layers |
| Canonical mock-256K, six allocated slots and slot 0 compared against GPU capture | Passed in 10m 02s |
| Existing prefill-runner unit tests | 10 passed |

Canonical mock-256K accuracy:

| Criterion | Required | Achieved | Result |
| --- | ---: | ---: | --- |
| Minimum per-head PCC | ≥ 0.91 | 0.911992 | PASS |
| Overall PCC | ≥ 0.97 | 0.973733 | PASS |
| Overall relative RMSE | < 0.232 | 0.229545 | PASS |

The worst head was layer 39, sliding V, head 9.
