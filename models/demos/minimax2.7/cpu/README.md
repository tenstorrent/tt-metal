# CPU reference demos

Reference / debugging entry points that run on plain PyTorch CPU — no ttnn,
no Tenstorrent device required. Intended for shape-checking, kernel
exploration, and producing golden activations to compare TT-NN modules
against.

## `demo.py` — random-weight forward pass

Instantiates MiniMax-M2.7 from its HF config (no checkpoint download) with a
**shrunk** architecture so it fits in CPU RAM, then runs a single forward
pass on random tokens.

```bash
# Online (small download: just the auto_map files)
python3 models/demos/minimax2.7/cpu/demo.py

# Offline (no network calls).
#   - transformers >= 4.57: uses the bundled configs/minimax-m2.7/config.json
#     directly via the native MiniMaxM2 classes.
#   - transformers <  4.57: uses the HF cache from a prior online run
#     (`~/.cache/huggingface/modules/...`); will fail if the cache is empty.
python3 models/demos/minimax2.7/cpu/demo.py --offline

# Bigger slice of the model (linearly more memory)
python3 models/demos/minimax2.7/cpu/demo.py --num-layers 4 --num-experts 16

# Just print the size estimate; do not allocate weights
python3 models/demos/minimax2.7/cpu/demo.py --check-only

# Build the full 62-layer / 256-expert model on the *meta* device
# (zero real allocation) and dump the nn.Module tree, then exit.
python3 models/demos/minimax2.7/cpu/demo.py --print-structure
```

`--print-structure` is the right tool when you want to inspect every layer
of the stock 229 B model without 458 GB of RAM. ``torch.device("meta")``
allocates zero bytes; only shape/dtype metadata is kept, which is enough
for ``print(model)`` and ``sum(p.numel() for p in model.parameters())``.
You **cannot** run a forward pass on a meta-device model — for that, use
the default mode (which shrinks).

### Building from local reference files

`../reference/` ships verbatim copies of MiniMax's `modeling_minimax_m2.py`,
`configuration_minimax_m2.py`, and `config.json`. When you pass
`--from-reference`, the demo builds the model strictly from those files via
`AutoConfig.from_pretrained(reference_dir, trust_remote_code=True)` — fully
offline, ignoring both the HF cache and any native `MiniMaxM2*` classes
that might be in your transformers install.

```bash
# Random-init forward pass driven by the reference modeling file
python3 models/demos/minimax2.7/cpu/demo.py --from-reference

# Print the full module tree using the exact reference modeling file
python3 models/demos/minimax2.7/cpu/demo.py --from-reference --print-structure
```

This is the right mode when you want to:
- Audit / hack the modeling code in `../reference/`.
- Pin to a specific HF commit by re-pulling those three files and committing them.
- Run on an air-gapped host that has never seen `MiniMaxAI/MiniMax-M2.7`.

If `../reference/` is missing files, the demo prints the exact
`curl` / `huggingface-cli download` command needed to populate it.

By default the script overrides the HF config to:

| Field | Stock M2.7 | Demo default |
|---|---|---|
| `num_hidden_layers` | 62 | 2 |
| `num_local_experts` | 256 | 8 |
| `num_experts_per_tok` | 8 | min(8, num_experts) |
| `quantization_config` | FP8 (e4m3, block 128×128) | None (random init) |

Result: ~1.5 B params, ~3 GB at bf16 — fits on a normal dev box.

See `python3 models/demos/minimax2.7/cpu/demo.py --help` for all flags.

### Dependencies

```bash
pip install -r models/demos/minimax2.7/requirements.txt
```

`transformers >= 4.57.0` is required because that is when the `minimax_m2`
model type was added natively (`MiniMaxM2ForCausalLM`,
`MiniMaxM2Config`). Older versions can still work in **online mode**
(`--offline` removed) since `trust_remote_code=True` will fetch the
modeling files from the HF repo.
