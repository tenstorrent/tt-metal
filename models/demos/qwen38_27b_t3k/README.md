# Qwen3.8-27B on T3K

TP8 implementation of `Qwen/Qwen3.8-27B` on a Wormhole B0 T3K, four n300 cards
as eight chips in a `(1, 8)` mesh. It supports prefill, traced decode, device
sampling and vLLM serving.

The Blackhole P300_X2 implementation is the sibling tree
`models/demos/qwen38_27b_qb2`; this one refuses any other mesh at construction.
The decoder policy below is still the QB2 TP4 measurement with a T3K overlay on
top, because that is where these kernels were tuned.

Collectives run as Ring, which requires the mesh to be opened with
`FabricConfig.FABRIC_1D_RING`; a linear fabric cannot route the wrap-around hop
and every collective hangs. Construction rejects that mismatch rather than
leaving it to a device timeout.

## Capacity

- Maximum supported context: 262,144 tokens on both meshes. For T3K the
  per-device budget is derived in `doc/context_contract.json`.
- Up to 16 concurrent users.
- Decode uses batch sizes 1, 8, and 16. Other active-user counts are padded to
  the next supported batch size: 2–8 use batch 8, and 9–16 use batch 16.

## Performance

Batch-1 vLLM performance on QB2/P300x2:

| Input / output tokens | Tokens/s/user | TTFT |
| --- | ---: | ---: |
| 128 / 128 | 39.4 | 67.9 ms |

Largest tested input sequence length on QB2: 261,892 tokens with 252 output
tokens.

Batch-1 performance on T3K, warmed, same shape:

| Input / output tokens | Tokens/s/user | TTFT |
| --- | ---: | ---: |
| 128 / 128 | 16.6 | 245 ms |

T3K reaches roughly a third of the QB2 decode throughput. It has 64 worker cores
against about 110, one usable ethernet link per chip pair against two, and one
DRAM reader per bank because multiple readers per bank are Blackhole-only.
`doc/multichip_evidence.md` carries the layer-stack breakdown and the remaining
gaps.

## Evaluation

No accuracy result has been established on this mesh. GPQA Diamond has been served end to end
once, scoring 84.85 over 198 samples against a published 89.2, but three credited samples commit
no answer and excluding any one of them fails the 0.95 ratio check. See the end-to-end eval
section of `doc/multichip_evidence.md`. Terminal-Bench and SWE-bench have not been run here.

## Run the demo

Build tt-metal and activate its Python environment:

```bash
./build_metal.sh --enable-ccache
source python_env/bin/activate
export PYTHONPATH="$PWD:$PWD/ttnn:$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B/snapshot
```

Run the model:

```bash
python models/demos/qwen38_27b_t3k/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

For vLLM serving, set `EXTRA_MODELS_DIR` to `models/demos`. The model is
registered as `TTQwen38ForCausalLM`.
