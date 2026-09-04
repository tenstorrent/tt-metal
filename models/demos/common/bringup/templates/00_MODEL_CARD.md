# Model card — <MODEL>

Every row needs a **Source**: a `config.json` key, a `path:line`, or a shown derivation.
A row you could not verify is marked `UNVERIFIED` and copied into `07_RISKS.md`.

## 1. Identity
| Field | Value | Source |
|---|---|---|
| HF repo / local path | | |
| How the identity was resolved | | |

## 2. Architecture
| Fact | Value | Source |
|---|---|---|
| architecture | | `architectures[0]` |
| layers | | `num_hidden_layers` |
| hidden | | `hidden_size` |
| FFN intermediate | | `intermediate_size` |
| activation | | `hidden_act` |
| Q heads / KV heads | | `num_attention_heads` / `num_key_value_heads` |
| head_dim | | derived: show the formula |
| norm + eps | | |
| RoPE theta / scaling | | see LANDMINES: not an attribute on transformers 5.x |
| vocab | | `vocab_size` |
| biases (attn / mlp) | | |

## 3. What this model does NOT have
> The anti-bloat control. The nearest templates have features yours does not; listing the absences
> is what stops them being copied in. One line each, e.g. no MoE, no attention sinks, no
> sliding-window, no QK-norm, no partial RoPE, no biases.

## 4. Deployment target
| Field | Value | Derivation |
|---|---|---|
| mesh shape | | |
| TP | | show the divisibility arithmetic **and** any hard constraint (e.g. a packed KV cache forcing TP == num_key_value_heads) |
| SP | | |
| chunk / max_seq_len constraints | | |
