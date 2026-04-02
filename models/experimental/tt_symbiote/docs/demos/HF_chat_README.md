# HF_chat.py

An interactive chatbot demo that runs the `inclusionAI/Ling-mini-2.0` model accelerated on a Tenstorrent T3K system using the TT-Symbiote framework.

## What It Does

`HF_chat.py` loads `inclusionAI/Ling-mini-2.0`, transparently replaces key PyTorch modules with TTNN-accelerated equivalents, and provides a terminal-based chat interface. The heavy compute (linear layers, activations, attention, normalization) runs on Tenstorrent devices while the rest of the model stays in PyTorch.

> **Note:** Only `inclusionAI/Ling-mini-2.0` is supported. The TTNN module replacements are specific to the Ling MoE architecture and will not work with other models.

## Prerequisites

- **Hardware**: Tenstorrent T3K
- **Software**: A working `tt-metal` environment with `ttnn` and `torch` installed
- **Python packages**: `transformers`, `tqdm`
- **Model access**: Internet access to download the model from HuggingFace on first run (or a local cache). The script uses `trust_remote_code=True`, which downloads and executes model code from the HuggingFace repository.

## How to Run

Ensure your `tt-metal` environment is set up with the following variables configured:

- `TT_METAL_HOME` — path to the `tt-metal` root
- `PYTHONPATH` — includes `TT_METAL_HOME`
- `WH_ARCH_YAML` — e.g. `wormhole_b0_80_arch_eth_dispatch.yaml`
- `ARCH_NAME` — `wormhole_b0`
- `MESH_DEVICE` — `T3K`

The `tt-metal` Python virtual environment must also be activated.

From the `tt-metal` project root:

```bash
python3 models/experimental/tt_symbiote/demos/HF_chat.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-new-tokens` | `256` | Maximum number of tokens generated per response |

## Chat Commands

Once the chatbot is running you will see a `You:` prompt. Type naturally to chat.

| Command | Effect |
|---------|--------|
| `/clear` | Resets conversation history and KV cache |
| `/clear_trace` | Releases compiled TTNN traces (frees device memory) |
| `exit` | Exits the chatbot |
| `Ctrl+C` | Also exits |

## What Happens at Startup

1. **Device setup** -- Opens a TTNN mesh device with a 1D ring fabric.
2. **Model loading** -- Downloads the HuggingFace model and tokenizer.
3. **Module replacement** -- Swaps PyTorch modules for TTNN-accelerated versions:
   - Decoder layers -> `TTNNBailingMoEDecoderLayerPadded` (which internally creates `TTNNBailingMoEAttention`, `TTNNSDPAAttention`, `TTNNMoERouterDecode`, `TTNNExperts`, and other sub-modules)
   - RMSNorm -> `TTNNDistributedRMSNorm`
   - `nn.Linear` -> `TTNNLinearIColShardedWRowSharded`
   - `nn.SiLU` -> `TTNNSilu`
4. **Weight preprocessing** -- Converts and transfers all weights to device memory.
5. **Warmup** -- Runs dummy inputs at sequence lengths 256 and 1024 to compile and trace TTNN kernels.
6. **Chat loop** -- Ready for interactive use.

Startup takes several minutes depending on model size and device count. The warmup step is especially important -- it ensures decode traces are pre-compiled so actual chat responses are fast.

## Limitations

- **Ling-mini-2.0 only**: The TTNN module replacements target the Ling MoE architecture. Other models are not supported.
- **Batch size 1 only**: The paged KV cache is configured for a single user session. There is no concurrent request support.
- **No streaming output**: Responses appear all at once after generation completes, not token-by-token.
- **Conversation history grows unbounded**: Every turn re-encodes the full chat history. Long conversations will eventually exceed the model's context window or run out of KV cache blocks (max 32 blocks of 64 tokens = 2048 tokens of cache).
- **Warmup is not optional**: Skipping or interrupting warmup will result in uncompiled traces and slower or broken inference.
- **Device memory**: Large models or long contexts may exhaust device SRAM/DRAM. There is no graceful handling if this occurs.
