# MTP eh_proj Weight Ordering Analysis

## Summary
**THE BUG: tt-metal glm4_moe (REAP-218B) has the WRONG split order for eh_proj weights.**

## Key Finding

The weight matrix `eh_proj` has shape `[hidden_size, 2*hidden_size]` in PyTorch/HuggingFace format (out_dim, in_dim).

The input concatenation in MTP is: `cat([embed, hidden])`
- First hidden dimensions → embed input
- Second hidden dimensions → hidden input

### Reference Implementations (ALL AGREE)

#### 1. GLM-4 MoE (vllm/vllm/model_executor/models/glm4_moe_mtp.py:113-115)
```python
hidden_states = self.eh_proj(
    torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
)
```
**Concat order: `[embed, hidden]`**

#### 2. DeepSeek-V3 MTP (vllm/vllm/model_executor/models/deepseek_mtp.py:109-111)
```python
hidden_states = self.eh_proj(
    torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
)
```
**Concat order: `[embed, hidden]`**

#### 3. MiMo MTP (vllm/vllm/model_executor/models/mimo_mtp.py:81-83)
```python
hidden_states = self.input_proj(
    torch.cat([previous_hidden_states, inputs_embeds], dim=-1)
)
```
**EXCEPTION: MiMo does `[hidden, embed]` — but MiMo is a different architecture**

#### 4. tt-metal glm4_moe_lite (Flash, WORKING) (model_tt.py:1505-1508)
```python
concat = ttnn.concat([enorm_out, hnorm_out], dim=3)
proj = ttnn.linear(concat, self.mtp_eh_proj_w)
```
**Concat order: `[embed, hidden]`** ✓ CORRECT

#### 5. tt-metal glm4_moe (REAP-218B, BROKEN) (model_tt.py:580-593)
```python
eh_proj_full = mtp_state[f"model.layers.{mtp_layer_idx}.eh_proj.weight"]  # [hidden, 2*hidden]
mtp_eh_proj_e_w = _linear_weight_tt(
    torch_weight_out_in=eh_proj_full[:, :hidden].contiguous(),  # [hidden, hidden]
    ...
)
mtp_eh_proj_h_w = _linear_weight_tt(
    torch_weight_out_in=eh_proj_full[:, hidden:].contiguous(),  # [hidden, hidden]
    ...
)
```
Then (model_tt.py:1219-1223):
```python
proj_e = ttnn.linear(enorm_out, self.mtp_eh_proj_e_w)
proj_h = ttnn.linear(hnorm_out, self.mtp_eh_proj_h_w)
proj = ttnn.add(proj_e, proj_h)
```

**PROBLEM: The split is BACKWARDS!**
- `mtp_eh_proj_e_w` uses `eh_proj_full[:, :hidden]` (FIRST half)
- `mtp_eh_proj_h_w` uses `eh_proj_full[:, hidden:]` (SECOND half)

But the WEIGHT NAMING says:
- `_e_w` = embed weight
- `_h_w` = hidden weight

### Correct Split Should Be

For `eh_proj` weight `[hidden, 2*hidden]`:
- Columns `[:, :hidden]` correspond to the **EMBED input** (first half of the concat)
- Columns `[:, hidden:]` correspond to the **HIDDEN input** (second half of the concat)

**CURRENT CODE ASSIGNMENTS (glm4_moe):**
```
mtp_eh_proj_e_w = eh_proj_full[:, :hidden]      # Embed weight (CORRECT BY ACCIDENT)
mtp_eh_proj_h_w = eh_proj_full[:, hidden:]      # Hidden weight (CORRECT BY ACCIDENT)
```

Wait, the current code is actually CORRECT!
- `eh_proj_full[:, :hidden]` → embed
- `eh_proj_full[:, hidden:]` → hidden

But then the computation:
```python
proj_e = ttnn.linear(enorm_out, self.mtp_eh_proj_e_w)      # enorm_out ⊗ eh_proj_e_w
proj_h = ttnn.linear(hnorm_out, self.mtp_eh_proj_h_w)      # hnorm_out ⊗ eh_proj_h_w
proj = ttnn.add(proj_e, proj_h)
```

IS ALSO CORRECT:
- `enorm_out` (embed normalized) × `eh_proj_e_w` (embed weight half)
- `hnorm_out` (hidden normalized) × `eh_proj_h_w` (hidden weight half)
- Sum them

So the **split itself is correct**. The issue must be elsewhere.

## Potential Issues

1. **Broadcasting/Batching**: The ttnn.linear calls might not be handling the batch dimensions correctly
2. **Dtype Mismatch**: BF8/BF16 conversion issues in the split weights
3. **Transpose Order**: The weight matrices might need a transpose or different layout
4. **Accumulation**: Adding two independent matmul outputs instead of a single concatenated matmul

## Recommendation

The split order appears CORRECT. Need to investigate:
- Weight cache (is it poisoned from a previous broken build?)
- ttnn.linear output dimensions and batch handling
- Whether adding two matmul outputs matches the original concat+linear behavior
- Comparison with eager mode (run the same Python split on CPU to verify)
