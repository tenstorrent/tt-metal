# Qwen3.8-27B

Tensor-parallel implementation of `Qwen/Qwen3.8-27B` supporting prefill, traced
decode, device sampling and vLLM serving. Two meshes are qualified, and the
tensor-parallel width follows the mesh:

| Hardware | Mesh | TP |
| --- | --- | ---: |
| Blackhole P300_X2 (QB2) | `(1, 4)` | 4 |
| Wormhole B0 T3K | `(1, 8)` | 8 |

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

No equivalent T3K numbers are published yet: the multichip and full-model
performance measurements for that mesh have not been taken. See the gaps in
`doc/multichip_evidence.md`.

## Evaluation

Evaluation results:

- GPQA Diamond: 9/10 (90%).
- Terminal-Bench 2.1: 4/5 (80%).
- SWE-bench Verified: 3/5 (60%).

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
python models/demos/qwen38_27b_qb2/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

For vLLM serving, set `EXTRA_MODELS_DIR` to `models/demos`. The model is
registered as `TTQwen38ForCausalLM`.
