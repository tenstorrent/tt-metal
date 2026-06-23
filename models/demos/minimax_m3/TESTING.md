# MiniMax-M3 prefill — running the tests

How to run the M3 prefill PCC tests on **single device** and on the **Blackhole Galaxy** (32 chips).
All tests compare TTNN modules against **self-authored torch references with random weights** (no HF
checkpoint needed) — see `PREFILL_PROPOSAL.md` §0.2b/§0.4 for the why.

## Environment (always)

```bash
cd /data/vmelnykov/tt-metal
export TT_METAL_HOME=/data/vmelnykov/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source python_env/bin/activate          # ttnn lives here; /usr/bin/python3 has no ttnn
```

> ⚠️ **Don't pipe a device run through `head`/`tail`** — it SIGPIPE-kills the process mid-run
> (and the pipeline's "exit 0" is `head`'s, not the test's). Redirect to a file and grep that:
> `python … > /tmp/run.log 2>&1; grep -iE "pcc|PASS|FAIL" /tmp/run.log`.

---

## 1. Single device (TP=1) — the unit suite

No mesh-graph-descriptor needed. The unit tests parametrize the mesh; `-k 1x1` selects single card.

```bash
pytest models/demos/minimax_m3/tests/unit/ -k 1x1 -q
```

Individual modules (each is a `test_*_vs_ref.py`):

| Test | Validates |
|---|---|
| `test_norm_vs_ref.py` | Gemma `(1+w)` RMSNorm |
| `test_swiglu_vs_ref.py` | clamped `swigluoai` SwiGLU |
| `test_qk_norm_vs_ref.py` | per-head QK-norm |
| `test_dense_mlp_vs_ref.py` | dense MLP (layers 0-2) |
| `test_attention_vs_ref.py` | full GQA attention block |
| `test_decoder_layer_vs_ref.py` | dense decoder layer + hybrid schedule |
| `test_moe_vs_ref.py` | MoE block (router + experts + shared + routed_scaling) |
| `test_moe_decoder_layer_vs_ref.py` | MoE (sparse) decoder layer |
| `test_model_vs_ref.py` | full model assembly → logits |

Example (one module): `pytest models/demos/minimax_m3/tests/unit/test_attention_vs_ref.py -k 1x1 -q`

---

## 2. Galaxy / multi-card (32 chips)

The box is a plain **8×4 MESH** (no torus) → multi-card needs a **mesh-graph-descriptor** + `FABRIC_1D`
+ CCL `Topology.Linear` (the harness/standalone scripts set fabric + linear topology for you; you
only set the MGD env). `MeshConfig` maps **tp = cols, ep = rows**.

```bash
# Stock single_bh_galaxy MGD is [8,4] -> opening (8,4) gives TP=4 (M3's tensor-parallel factor).
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
```

### 2a. TP=4 (attention + dense MLP CCL collectives) — `(8,4)`, `-k 8x4`

On `(8,4)`: tp=cols=4, the 8 rows replicate. Exercises the o_proj reduce-scatter/all-gather and the
dense-MLP down all-reduce.

```bash
pytest models/demos/minimax_m3/tests/unit/test_attention_vs_ref.py \
       models/demos/minimax_m3/tests/unit/test_dense_mlp_vs_ref.py \
       -k 8x4 -q
```

### 2b. EP=32 MoE (expert-parallel) — standalone script, `(8,4)`

128 experts / top-4 → 4 experts/device across 32 chips. Reuses deepseek dispatch/combine + the M3
`CompositeRoutedExpert` (clamped swigluoai). Standalone (no pytest); redirect output to a file:

```bash
python models/demos/minimax_m3/tests/test_ep_moe_vs_ref.py --rows 8 --cols 4 --seq 128 \
    > /tmp/ep_moe.log 2>&1
grep -iE "ep-moe|pcc|PASS|FAIL" /tmp/ep_moe.log
```

### 2c. SP=8 building blocks (op + cache) — `(8,4)`, `-k 8x4`

De-risk tests for the SP=8 + chunked-KV rearchitecture (PREFILL_PROPOSAL §0.2d/§6.6). These validate
the pieces **in isolation** — they are NOT yet wired into the model attention path. **Both require the
local `ring_joint` grouped-V kernel fix** (branch carries it; `ring-joint-gqa-v` is prepped for a PR).

```bash
# SP attention op: ring_joint GQA causal, SP=8 x TP=4, vs torch causal golden (PCC ~0.99998)
pytest models/demos/minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py -k 8x4 -q -s

# GQA chunked-KV cache: TP-sharded heads + SP block-cyclic seq, write/readback vs torch (PCC ~0.99997)
pytest models/demos/minimax_m3/tests/unit/test_kv_cache_gqa_sp_vs_ref.py -k 8x4 -q -s
```

### 2d. Real-weights ground-truth (needs the 869 GB checkpoint + HF_MODEL)

```bash
# HF minimax_m3_vl golden vs our TTNN bf4 galaxy run (first-token 8/8 + oracle). CPU-offload, slow.
export HF_MODEL=/path/to/MiniMax-M3
uv run --no-project --with "transformers>=5.12" --with torch --with accelerate --python 3.10 \
    python models/demos/minimax_m3/tests/golden_hf_first_token.py
# Our real-weights generation (galaxy): galaxy_first_token_m3.py / galaxy_generate_m3.py (see file headers).
```

---

## 3. Numerical oracle — correctness vs the real `minimax_m3_vl` (no device, no download)

Our `*_vs_ref` tests compare TTNN against *self-authored* torch refs (we wrote both sides). This
script closes the loop by checking those refs against the **upstream `transformers minimax_m3_vl`**
(needs transformers ≥5.12, which our ttnn venv doesn't have, and we don't bump the repo pin).

It's a **standalone script, not a pytest test** (no `test_` prefix → the pytest suite skips it). Run
it in a throwaway env — no touch to `python_env`, no checkpoint, CPU only. **Option A (uv):**

```bash
uv run --no-project --with "transformers>=5.12" --with "torch" --python 3.10 \
    python models/demos/minimax_m3/tests/oracle_minimax_m3_vl_pcc.py
```

**Option B (no uv):**

```bash
python3.10 -m venv /tmp/m3_oracle && /tmp/m3_oracle/bin/pip install -q "transformers>=5.12" torch
/tmp/m3_oracle/bin/python models/demos/minimax_m3/tests/oracle_minimax_m3_vl_pcc.py
```

First run pulls transformers + torch (~GB, then cached). Expect `PCC = 1.000000` on RMSNorm /
DenseMLP / SparseMoeBlock / Attention (both fp32, same random weights). Together with the device
tests: `TTNN ==(device PCC)== our refs ==(this, exact)== real M3`.

---

## Notes / gotchas

- **Why `(8,4)` for TP=4 and not `(1,4)`:** the stock MGD is `[8,4]`; `(8,4)` gives tp=4 with the 8
  rows replicating, which matches the real galaxy physical mapping (TP on the 4-axis, the 8-axis is
  SP/EP). A clean isolated `(1,4)` would need a custom 1x4 MGD (not currently added).
- **MoE multi-card == EP** (not replicated TP): at >1 device the MoE auto-switches to the
  expert-parallel path, which requires `num_experts % num_devices == 0`. So use the EP script (2b),
  not the `(8,4)` unit MoE test.
- **SP=8 / 1M context:** not runnable yet — we use full-GQA as the MSA placeholder, so prompts are
  capped at **S ≲ 2048** (where GQA == MSA). SP matters only once MSA + long context land.
- **Pre-commit:** if a commit silently rolls back ("stashed changes conflicted"), run
  `pre-commit run --files <changed files>` to apply formatting, then `git add` + `git commit --no-verify`.
