# Reference knowledge

Worked references a planner or engineer reads when its step needs them (the repo map points here; nothing in this
folder is read by every step). Each reference keeps an HTML page for people and a text version for agents; claude.ai
artifact links are not readable by agents or by other people, so a reference lives here, not as a link.

| Reference | What it is |
|---|---|
| `kimi_k27_4x4_layer.md` (`.html`) | One Kimi K2.7 layer (MLA + 384-expert MoE) on a 4x4 mesh, SP = 4 rows x TP = 4 columns: what each chip holds per step, the collectives, and the tt-moe-nappkin cost per step. The layout xing40_a4b_d_p uses with 2 columns. |
