---
name: bringup
description: Automated end-to-end model bring-up on Tenstorrent hardware. Orchestrates Architecture → Reference → TTNN → Verification → Server phases with unsupervised execution, per-op debug loops, and escalation after 10 attempts. Use /bringup <HF_link|model_path> to start, /bringup <model_name> to resume.
---

# SKILL: Automated Model Bring-Up

## Entry Points

```bash
# New model
/bringup https://huggingface.co/org/ModelName-7B
/bringup /path/to/local/pytorch/model

# Resume existing (agent reads Current Status from BRINGUP_LOG.md)
/bringup ModelName-7B
/bringup models/demos/molmo2
```

On resume: read `## Current Status` from `BRINGUP_LOG.md` → continue from exact phase/block/attempt.

---

## Pre-Flight: Repo-Wide CPU Audit

Before any phase starts, audit ALL existing models for CPU usage patterns:

```bash
grep -rn "F\.linear\|torch\.matmul\|\.cpu()\|to('cpu')\|F\.softmax\|F\.gelu" \
    models/demos/*/tt/*.py models/tt_transformers/tt/*.py \
    | grep -v "__init__\|#\|test_\|reference/"
```

Build a classification table:

| Pattern | Models | Location | Verdict |
|---------|--------|----------|---------|
| F.linear in forward_prefill | molmo2, qwen3_vl | lm_head only | DOCUMENTED |
| CPU gather for pooling | molmo2 | preprocessing | ACCEPTABLE |
| ... | | | |

**Classifications:**
- `ACCEPTABLE` — init, preprocessing, one-time upload
- `DOCUMENTED` — known tradeoff with explanation
- `SHORTCUT` — was workaround, must not repeat
- `BLOCK` — in forward inference pass, no justification → agent will not reproduce

This table is the reference for CPU decisions in the new model's bringup.

---

## CPU Rules (enforced, not advisory)

```
ALLOWED:
  __init__(): weight loading, tensor conversion, buffer creation
  Preprocessing: pooling index construction, patching, tokenization, bicubic interp
  One-time upload: positional embeddings, transformation matrices

BLOCKED in forward() / forward_prefill() / forward_decode():
  F.linear, F.softmax, F.gelu, torch.matmul, torch.einsum
  Boolean indexing to replace TTNN ops
  Any ttnn.to_torch() followed by computation

If agent is about to write blocked CPU code:
  → REFUSE, log "CPU blocked: searching TTNN alternative"
  → Use as evidence for next debug attempt
  → If no TTNN alternative exists after 10 attempts: escalate (do NOT use CPU)
```

---

## Phase 1: Architecture

**Runs on:** CPU | **Parallelism:** Single agent

### Steps

1. **Read all HF artifacts** (ALL files, not just config.json):
   - Every `*.json` config file
   - Every `*.py` source file (modeling, processing, configuration)
   - Weight index for component inventory

2. **Search current `models/` for reusable implementations**:
   ```bash
   # Sort by most recently modified
   git log --format="%ai %f" -- models/demos/*/tt/*.py models/tt_transformers/tt/*.py \
       | sort -r | head -30
   ```
   Score by: (a) same op types, (b) same device target (N150/T3K/Galaxy), (c) same parallelism pattern. Pick highest-scoring match per block.

3. **Find unit test for every op planned**:
   ```bash
   find tests/ttnn/unit_tests/operations/ -name "test_*.py" | xargs grep -l "<op_name>"
   ```
   Every op must have a unit test reference before implementation starts.

4. **Divisibility checks** for target device:
   ```python
   for name, val in [("n_q", n_q), ("n_kv", n_kv), ("intermediate", I)]:
       assert val % n_devices == 0, f"{name}={val} not divisible by {n_devices}"
   ```

5. **Memory budget**: weights + KV cache < 85% device DRAM

### Gate: Architecture is COMPLETE when
- [ ] All weight-file components mapped to TTNN impl or flagged NEW
- [ ] All non-standard ops have TTNN strategy (or escalated)
- [ ] Every planned op has a unit test found
- [ ] Divisibility checks pass for target device
- [ ] Memory budget < 85%
- [ ] ARCHITECTURE.md written with machine-parseable block inventory table
- [ ] MESH_SHARDING_PLAN.md written with one row per block
- [ ] Every divergence from the canonical Galaxy port has a documented reason

### Output: MESH_SHARDING_PLAN.md (mandatory before Phase 3)

One row per block (attention / MLP / norm / LM head). For each:
- `cluster_axis` for every CCL op, head-sharding axis, output memcfg, num_links
- File-and-line citation to the canonical Galaxy port row this matches
  (e.g. `llama3_70b_galaxy/tt/llama_attention.py:N`)
- For any deviation from the canonical port: one-line reason (e.g. "head count
  not divisible by mesh.rows=8")

**Two-reference disambiguation.** When both a same-model v1 (correctness-locked)
and a canonical Galaxy port (perf-locked) exist:
- v1 is the **math oracle** (HF key map, qknorm placement, rope_dim, gate placement).
- Canonical Galaxy port is the **structural oracle** (cluster_axis, num_links,
  head-sharding axis, output memcfg, residual-stream dtype).
- On disagreement, default to the Galaxy port unless this plan documents the
  deviation with a reason. v1's structural choice may have been correctness-locked
  at one ISL, not perf-locked.

Acceptance gate for Phase 1: this file exists AND every row either matches the
canonical port OR documents the deviation. **Phase 3 cannot start without it.**

### Outputs
```
models/demos/{model}/
├── ARCHITECTURE.md          # block inventory, weight mapping, TP plan, bottlenecks
├── MESH_SHARDING_PLAN.md    # per-block sharding parity table vs canonical Galaxy port
```

---

## Phase 2: Reference

**Runs on:** CPU only | **Parallelism:** All blocks in parallel (CPU, no device needed)

### Steps

1. **Build `load_checkpoint.py`**:
   Location: `models/demos/{model}/tt/load_checkpoint.py`
   Maps HF checkpoint keys → weight dict used by reference and TTNN.
   Pattern: see `models/demos/qwen3_vl/tt/load_checkpoint.py` for reference.

2. **Generate `reference/functional.py`** for each block with capture_intermediates:
   ```python
   def attention_forward(x, state_dict, cos, sin, capture_intermediates=False):
       caps = {}
       qkv = F.linear(x, state_dict["wqkv"])
       if capture_intermediates: caps["qkv_proj"] = qkv.clone()
       # ... every op gets a capture point
       return output, caps   # caps is {} when capture_intermediates=False
   ```

3. **Verify against HF model** — two gates:
   ```python
   # Gate 1: PCC
   pcc = torch.corrcoef(torch.stack([ref.flatten(), hf.flatten()]))[0,1].item()
   assert pcc > 0.99

   # Gate 2: Element-wise (bfloat16 tolerance)
   p99_diff = (ref - hf).abs().flatten().kthvalue(int(0.99 * n)).values.item()
   assert p99_diff < 0.02
   ```

4. **Save persistent golden tensors** — one file per (layer, op_boundary):
   ```
   reference/golden/
     layer{N}_attention_qkv_proj.pt
     layer{N}_attention_rope_q.pt
     layer{N}_attention_sdpa.pt
     layer{N}_attention_output_proj.pt
     layer{N}_mlp_gate_proj.pt
     ...
     integration_layer{N}_output.pt   # for integration test binary search
   ```

### Gate: All blocks pass PCC > 0.99 AND p99_diff < 0.02 vs HF

### Outputs
```
models/demos/{model}/
├── tt/load_checkpoint.py
├── reference/functional.py
└── reference/golden/*.pt
```

---

## Phase 3: TTNN

**Runs on:** Target device (N150 / T3K / Galaxy) | **Parallelism:** SERIAL (device occupied during test)

### For each block (serial order: simpler → complex):

#### 3a. Find reusable implementation
```python
# Score existing implementations:
# +3: same device target
# +2: same parallelism (TP8, DP, replicated)
# +2: same op types (GQA, SwiGLU, RMSNorm)
# +1: most recently modified
# Pick highest score; if delta < 30% → adapt; if > 70% → write new
```
Log: `"attention.py: adapted from qwen3_vl/tt/attention.py (RoPE style only)"`

**When both a same-model v1 and a canonical Galaxy port exist:**
- v1 is the **math oracle** — take from it: HF key map, qknorm placement,
  rope_dim, gate placement, custom-op math.
- Canonical Galaxy port is the **structural oracle** — take from it
  unconditionally: cluster_axis, num_links, head-sharding axis, output memcfg,
  residual-stream dtype, CCL buffer key conventions.
- If they disagree on a structural choice, the v1's choice is treated as
  potentially correctness-locked-not-perf-locked. Default to the Galaxy port
  unless `MESH_SHARDING_PLAN.md` says otherwise with a reason.

Log: `"attention.py: math from qwen3_6_galaxy/tt/llama_attention.py, sharding
from llama3_70b_galaxy/tt/llama_attention.py (cluster_axis=0 WO, num_links=2)"`

#### 3b. Implement (no CPU in forward)
Every TTNN block gets `debug_mode=False` parameter for op-level capture.

#### 3c. Run PCC test
```bash
pytest models/demos/{model}/tests/test_{block}_pcc.py -v -s
```
Agent runs directly. Must pass PCC > 0.99 AND p99_diff < 0.02.

#### 3d. Debug loop (max 10 attempts per block)

**On failure:**

```
Step 1: LOCATE — find first failing layer
  Binary search layers until layer K (first fail with real data flow)

Step 2: ISOLATE — prove it's layer K's bug, not compound
  Run layer K with golden input → if passes: compound error → search earlier
  Run layer K with golden input → if fails: real bug in layer K

Step 3: DRILL DOWN — find exact failing op within layer K
  Run block K in debug_mode=True → compare in-memory TTNN caps vs persistent golden
  → first op where PCC drops or p99_diff > threshold = failing op

Step 4: PROVE — write isolation unit test
  Location: models/demos/{model}/tests/test_{block}_op_isolation.py
  Uses exact shapes from the failing model, tests the single op
  Compares against tests/ttnn/unit_tests/operations/ reference usage

Step 5: HYPOTHESIZE — from tensor evidence, not intuition
  "rope_q: p99_diff=0.12, unit test shows HF uses rotary_embedding not rotary_embedding_llama"

Step 6: FIX — single targeted change

Step 7: LOG — append to BRINGUP_LOG.md (see format below)
```

**Compound error path:**
- Per-block test uses golden input → passes even with compound error
- Integration test uses real sequential data → exposes compound error
- Binary search using `integration_layer{N}_output.pt` golden tensors
- First layer M where real-data output diverges from golden = source

**After 10 attempts:** write escalation report, STOP, wait for hint.
Hint (free text or reference to working impl) resets counter to 0.

#### 3e. Per-block gate
- [ ] PCC > 0.99 AND p99_diff < 0.02 (isolated, golden input)
- [ ] Integration test passes (real sequential data, all layers together)
- [ ] **Per-device weight footprint matches the parallelization plan from ARCHITECTURE.md.**
      Run block `__init__`, sum `ttnn.Tensor` bytes per device. If a matmul weight
      uses `ReplicateTensorToMesh` when the plan said sharded → reject; rewrite
      with `ShardTensor2dMesh` + the necessary CCL ops in forward. Compare against
      the per-block budget in 7c. **This check must run before scaling layer count**
      — single-block tests cannot expose replicated-weight OOMs.
- [ ] **Sharding parity check passes.** Open `MESH_SHARDING_PLAN.md`, verify the
      row for this block matches the code as written (cluster_axis, num_links,
      head-sharding axis, output memcfg). Any drift from the plan must be
      explicitly justified before merge, not retro-fixed.
- [ ] **CCL op diff vs canonical port is empty (or annotated).** Grep this block's
      file for `cluster_axis=`, `num_links=`, `reduce_scatter`, `all_gather`,
      `line_all_reduce`. Compare to the same grep in the canonical Galaxy port.
      Every divergence is annotated in `MESH_SHARDING_PLAN.md`.
- [ ] **Full-layer-count smoke load.** After 1-layer PCC passes, instantiate the model
      with all `n_layers` decoder layers (DRAM only, no forward) on real target hardware.
      If model construction OOMs, the per-block budget is wrong — fix sharding now,
      not after the demo test fails.

### Integration Test (mandatory, runs after all blocks pass per-block)

```python
# Run full model forward with real sequential input
out_ttnn = ttnn_model.forward(input_ids, pixel_values, ...)
out_ref  = ref_model.forward(input_ids, pixel_values, ...)

# Check at every layer boundary using saved integration goldens
for layer_idx in range(n_layers):
    ttnn_layer_out = ttnn_model.get_layer_output(layer_idx)  # from debug capture
    ref_layer_out  = torch.load(f"reference/golden/integration_layer{layer_idx}_output.pt")
    assert verify(ttnn_layer_out, ref_layer_out)["passed"]
```

### Outputs
```
models/demos/{model}/tt/
├── model_config.py
├── load_checkpoint.py
├── attention.py
├── mlp.py
├── vision_block.py      (VLM)
├── vision_encoder.py    (VLM)
├── image_pooling.py     (VLM)
├── image_projector.py   (VLM)
├── prefill_mask.py      (VLM)
└── model.py

models/demos/{model}/tests/
├── test_tt_text_decoder.py       # PCC tests (8 standard blocks)
├── test_{block}_op_isolation.py  # auto-generated during debug
└── test_integration.py           # full model integration
```

---

## Phase 4: End-to-End Verification

**Runs on:** Target device

### Use existing prompt sets (ISL-matched)

```bash
# Find prompt sets from similar model type
ls models/demos/*/demo/sample_inputs/   # text_only.json, image_demo.json, etc.
```

For each prompt set (text-only, image, video):
1. Run reference model → capture output tokens
2. Run TTNN model → capture output tokens
3. Compare: exact token match OR first-char match (for MCQ) OR PCC on logits

### Gate: TTNN output matches reference for all prompts

---

## Phase 5: Server Integration

**Runs on:** Target device via tt-inference-server

See `/tt-inference-server` skill for full details. Summary:

1. Identify model type → pick generator_vllm.py template
2. Build `tt/generator_vllm.py` (no decode trace for variable-S servers)
3. Register in `tt-vllm-plugin/__init__.py`
4. Add `DeviceModelSpec` to `workflows/model_spec.py`
5. Start server + run test suite
6. Gate: accuracy ≥ reference - 2pp

---

## BRINGUP_LOG.md Format

```markdown
# {Model} Bringup Log

## Current Status
Phase: TTNN | Block: text_attention | Attempt: 3/10 | Status: IN PROGRESS
Next: text_mlp (after text_attention passes)

---

## 2026-05-01

### TTNN — text_attention — IN PROGRESS

**Attempt 1** 09:15
- Source: adapted from models/demos/qwen3_vl/tt/attention.py
- Result: PCC=0.887, p99_diff=0.12
- Failing op: rope_q (all prior ops ✓)
- Evidence: unit test shows HF uses rotate_half style, impl uses rotary_embedding_llama
- Hypothesis: RoPE style mismatch

**Attempt 2** 09:42
- Change: switched to ttnn.experimental.rotary_embedding with cat([cos,cos]) format
- Result: PCC=0.934, p99_diff=0.08
- Failing op: rope_q still (narrowed: cos/sin format correct, head reshape wrong)

**Attempt 3** 10:05
- Change: reshape to [1, n_heads, 1, head_dim] before RoPE, back after
- Result: ...

---

## 2026-04-30

### Reference — ALL BLOCKS — PASS
- text_attention: PCC=0.9999, p99_diff=0.001
- text_mlp: PCC=0.9999, p99_diff=0.001
- vit_encoder: PCC=0.9993, p99_diff=0.003

### Architecture — COMPLETE
- Reference: models/demos/qwen3_vl/tt/ (T3K TP8, GQA, SwiGLU)
- Non-standard ops: additive image injection (scatter-add) — strategy documented
- Memory budget: 6.2GB / 12GB per device ✓
```

---

## File Layout After Complete Bringup

```
models/demos/{model}/
├── ARCHITECTURE.md
├── README.md
├── demo/
│   ├── demo.py
│   └── sample_inputs/
├── reference/
│   ├── functional.py
│   ├── test_functional.py
│   └── golden/
│       ├── layer0_attention_qkv_proj.pt
│       ├── layer0_attention_rope_q.pt
│       └── integration_layer0_output.pt  (... per layer per op)
├── tt/
│   ├── load_checkpoint.py
│   ├── model_config.py
│   ├── attention.py
│   ├── mlp.py
│   ├── model.py
│   ├── generator_vllm.py
│   └── (vision files if VLM)
└── tests/
    ├── test_tt_text_decoder.py
    ├── test_integration.py
    └── test_*_op_isolation.py  (auto-generated during debug)
```

---

## Escalation Protocol

After 10 failed attempts on any block:

```
ESCALATION REPORT — {model} — {block} — {date}

All 10 attempts failed. Current state:
  Phase: TTNN | Block: text_attention

Attempt history:
  1. [09:15] Adapted qwen3_vl → PCC=0.887 — rope_q failing
  2. [09:42] Fixed RoPE style → PCC=0.934 — still rope_q
  ...

Current best hypothesis:
  The QK-norm interacts with the RoPE in a way not seen in other models.
  Specifically: after QK-norm, values are in range [-0.1, 0.1] which causes
  RoPE to be numerically insignificant. Reference shows values should be [-2, 2].

Suggested next steps:
  A. Check if QK-norm is applied before or after head reshape (line 47 vs line 52)
  B. Verify q_norm weight shape — may need per-head vs global norm

STOPPED — provide hint to continue (resets attempt counter to 0)
```

User provides hint → agent incorporates → resets counter → continues.
