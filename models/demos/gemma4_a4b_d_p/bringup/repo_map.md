# Repo map: gemma4_a4b_d_p

| Need | Where it is |
|---|---|
| Serving KV contract, mixed KV shapes per layer | `models/demos/gemma4_a4b_d_p/tt/runners/kv_contract.py` (one per-chip slab on the gpt_oss_d_p GQA substrate) + adapter-side read-back in `tt/runners/adapter.py` |
