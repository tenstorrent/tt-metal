# GLM-5.3-Flash text demo

`text_demo.py` runs real prompts through the resident mesh model and reports what the other tt-metal demos
report: TTFT per user (prefill + head + first-token readback), decode tok/s per user and aggregate (traced
decode step with the greedy ids fed back), the generated text per user, and a metrics JSON under
`generated/glm53_flash_demo/`.

| id | prompts | users per row | max seq len | generated |
|---|---|---|---|---|
| `questions_8-4users` | `sample_prompts/questions_8.json` (8 short questions) | 4 | 1024 | 64 |
| `prefill_128-4users` / `-32users` | tt_transformers `input_data_questions_prefill_128.json` (128-token prompts) | 4 / 32 | 1024 | 128 |
| `long_1k-4users`, `long_4k-4users` | tt_transformers 1 k / 4 k prompts | 4 | 2048 / 8192 | 64 |

Mesh and dtypes come from the environment: `GLM53_MESH_SHAPE` (2,4 block or 8,4 Galaxy), `GLM53_MODEL_EXPERT_DTYPE` /
`GLM53_MODEL_EXPERT_DOWN_DTYPE` (bfp8 | bfp4; bfp8 experts need the 8x4, a 2x4 block holds 45 layers only with bfp4),
`GLM53_DEMO_LAYERS` (default 0-44), `GLM53_MODEL_TRACE_MB` (512). Chip sets and descriptors for the blocks are in
`doc/mesh/README.md`.

Known limitation (sequence-parallel prefill v1): a prompt prefilled into slot `u` lands in slot `u` of every row, so a
multi-row mesh decodes `users_per_row` distinct prompts replicated over the rows; the aggregate rate counts every slot.
