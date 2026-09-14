# Qwen3-8B with long prompts on a Tenstorrent n300

How to run Qwen3-8B on one n300 card with prompts of 32,768 or 65,536 tokens, tuned for
generating text one token at a time. Every number here was measured on that hardware.

Engineering detail — what was tuned, what was tried and rejected — is in
[QWEN3_8B_LONG_CONTEXT_ENGINEERING_NOTES.md](QWEN3_8B_LONG_CONTEXT_ENGINEERING_NOTES.md).
You do not need it to run the model.

## What you get

Per generated token, 32,768-token prompt, one user, wall clock:

| | Before tuning | After tuning |
|---|---|---|
| Time per token | 36.17 ms | **28.73 ms** |
| Tokens per second | 27.6 | **34.8** |

That is 20.6% faster. Answer quality is unchanged: 85.16% agreement with a full-precision
reference on the first choice, 97.85% within the top five.

## Requirements

| | |
|---|---|
| Card | One n300 (two Wormhole chips) |
| OS / Python | Ubuntu 22.04, Python 3.10 |
| Disk | ~35 GB (16 GB of weights plus caches) |
| Network | Once, to download the weights |
| Account | A free HuggingFace account, for the weights |

## Install

1. **Driver and firmware** — one script, then reboot:
   ```bash
   curl -fsSL https://github.com/tenstorrent/tt-installer/releases/download/v2.1.0/install.sh -O
   chmod +x install.sh && ./install.sh --install-container-runtime=no
   ```
   `tt-smi` should then list your board.
2. **TT-Metalium** — `pip install ttnn`, then clone this repository and install the model
   requirements. Full options (container image, source build) in [INSTALLING.md](../../INSTALLING.md).
   ```bash
   git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
   pip install -r tt_metal/python_env/requirements-dev.txt
   sudo cpupower frequency-set -g performance
   ```
3. **Weights** — `huggingface-cli login`, then set the model name (below). The 16 GB download
   happens automatically on first run.

Check: `python -c "import ttnn; print(ttnn.get_device_ids())"` prints a list of device ids.

## Flags

The tuning itself needs **no flags** — it is keyed on the model's name inside TT-Metalium and
switches on by itself for Qwen3-8B. What you must set is only about *your machine*:

| Flag | Required? | What it does |
|---|---|---|
| `HF_MODEL=Qwen/Qwen3-8B` | **Yes** | Selects the model. |
| `PYTHONPATH=<tt-metal-dir>` | Yes, unless your shell is inside the repository | Lets Python find `models/`. |
| `TT_VISIBLE_DEVICES=0` | Only if the host has more than one board | Pins the run to one n300. Without it, a multi-card host spreads the model over every chip. Harmless on a single-card host. |
| `--page_block_size N` | No (default 256) | KV-cache tokens per block. 256 measured fastest for long prompts; smaller values save memory on many short conversations at no measured speed cost (64 vs 256: 28.71 vs 28.75 ms/token). Multiple of 32, power of two up to 256. |

## Run the long-context demos

Everywhere below, replace `<tt-metal-dir>` with the directory you cloned the repository into
— for example `/localdev/jerrywang/tt-metal`. It is a placeholder, not a real path; copying
it literally gives `No such file or directory`.

From inside the repository:

```bash
cd <tt-metal-dir> && export PYTHONPATH=$(pwd)

# 32k prompt, all three user counts (1, 2, 4) — about 4 minutes
TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
  pytest models/tt_transformers/demo/long_context_demo.py -s -k 32k

# 64k prompt — about 4½ minutes; the 4-user row is skipped (KV cache would exceed 8 GB per chip)
TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
  pytest models/tt_transformers/demo/long_context_demo.py -s -k 64k
```

From anywhere else, give the full path instead; `pytest` finds the repository's configuration
from it:

```bash
export PYTHONPATH=<tt-metal-dir>
TT_VISIBLE_DEVICES=0 HF_MODEL=Qwen/Qwen3-8B \
  pytest <tt-metal-dir>/models/tt_transformers/demo/long_context_demo.py -s -k 32k
```

To run a single row, name it. `b` is the number of concurrent users:

```
-k 32k-b1    -k 32k-b2    -k 32k-b4    -k 64k-b1    -k 64k-b2
```

No `-k` at all runs everything. The first run is much longer because it downloads the weights.

## What you should see

One summary table at the end of each run. These are the actual tables from one n300, from
exactly the two commands above, on 2026-09-14. Yours should land within a few tenths of a
millisecond of the `decode ms` column and within a few percent on `TTFT ms`.

`-k 32k` — 3 minutes 59 seconds:

```
ctx    users block prompt tok  build s    TTFT ms  compile ms  decode ms  tok/s/u   tok/s  rope        mode   status
32k        1   256      32768      5.4   12885.55       30.56      28.68     34.9    34.9  native      bench  ok
32k        2   256      32768      5.7   25862.46       36.17      34.84     28.7    57.4  native      bench  ok
32k        4   256      32768      7.3   51884.31       48.37      46.90     21.3    85.3  native      bench  ok
```

`-k 64k` — 4 minutes 36 seconds; the 4-user row is skipped before anything is built
(`KV cache needs 9.6 GiB per chip, over the 8 GiB budget`):

```
ctx    users block prompt tok  build s    TTFT ms  compile ms  decode ms  tok/s/u   tok/s  rope        mode   status
64k        1   256      65536      6.4   39149.41       36.46      34.87     28.7    28.7  yarn x1.61  bench  ok
64k        2   256      65536      7.3   78446.51       48.15      46.93     21.3    42.6  yarn x1.61  bench  ok
```

Reading the columns:

- **`decode ms`** — milliseconds per generated token, the number the tuning improves.
  **`tok/s/u`** is the same figure as a per-user rate; **`tok/s`** is all users combined.
- **`TTFT ms`** — time to first token: reading the whole prompt before anything comes out.
  Prompts are read one user at a time, so this grows in proportion to the user count:
  12.9, 25.9, 51.9 seconds for 1, 2, 4 users at 32k. The tuning does not address it.
- **`rope`** — `native` means the prompt fits the model's trained 40,960-token window.
  `yarn x1.61` means it was stretched past that with YaRN. See limit 1 below.
- **`compile ms`** — one-off cost on the first token, excluded from `decode ms`. Ignore it.

Two patterns worth knowing before you size a deployment. Each extra concurrent user costs
about 6 milliseconds per token at 32k (28.68 → 34.84 → 46.90) while combined throughput
keeps rising (34.9 → 57.4 → 85.3 tokens per second), so more users is more total work done
at a slower rate each. And doubling the prompt from 32k to 64k at the same user count costs
about the same as doubling the users at the same prompt: one user at 64k (34.87) matches
two users at 32k (34.84), and two at 64k (46.93) matches four at 32k (46.90).

## Known limits

1. **Randomised sampling runs on the host, not the card.** Greedy decoding (temperature 0)
   picks the next token on the card. A temperature above zero cannot, on this model, and is
   served correctly from the host at a cost: 38.4 vs 21.8 ms/token on a 1,024-token prompt.
   The demos are greedy. For vLLM this fallback is not yet verified — test before relying on it.
2. **TTFT hasn't specifically been optimized.**

## Other ways to run it

- **Your own prompt:** `simple_text_demo.py` accepts `--input_prompts <file.json>`. Add
  `--max_seq_len 40960` (its long-context case asks for 65,536, above this model's window, and
  skips otherwise) and keep `performance` in the `-k` filter. It uses 64-token blocks and
  cannot go past 40,960 tokens.
- **As an HTTP service:** the [Tenstorrent vLLM plugin](https://github.com/tenstorrent/vllm/blob/dev/plugins/vllm-tt-plugin/README.md).
  Set its block size to 256 and prefer greedy requests.

## Help

[Discord](https://discord.gg/tenstorrent) · [Issues](https://github.com/tenstorrent/tt-metal/issues)
(include `tt-smi` output, the exact command, the full error) · [INSTALLING.md](../../INSTALLING.md)
