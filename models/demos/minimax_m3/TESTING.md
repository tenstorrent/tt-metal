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
