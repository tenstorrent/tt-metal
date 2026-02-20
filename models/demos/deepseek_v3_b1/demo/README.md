# DeepSeek V3 B1 Demo CLI

This folder contains a CLI for running the current DeepSeek V3 B1 demo.

The demo runs prefill + decode over `DeepSeekV3` sockets and streams decoded text to stdout.

## Requirements

Assuming your environment is configured correctly:

1. Enable slow dispatch mode:

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
```

2. Run the demo:

```bash
python -m models.demos.deepseek_v3_b1.demo.cli --prompt "Once upon a time" --max-new-tokens 128
```
