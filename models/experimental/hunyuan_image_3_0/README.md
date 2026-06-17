# HunyuanImage-3.0 (tt-metal)

Experimental TTNN port of [Tencent HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0).

## Tokenizer

Host-side tokenizer code and assets live under `ref/tokenizer/`:

| Path | Description |
|------|-------------|
| `ref/tokenizer/hunyuan_tokenizer.py` | Public API (`HunyuanTokenizer`) |
| `ref/tokenizer/gen_image_inputs.py` | Host preprocess bundle for device upload |
| `ref/tokenizer/assets/config.json` | Model config used by the tokenizer stack |
| `ref/tokenizer/assets/tokenizer_config.json` | HF tokenizer config |
| `ref/tokenizer/assets/tokenizer.json` | BPE vocab (~24 MB; not in git) |

Download `tokenizer.json` (and refresh `tokenizer_config.json`) from Hugging Face:

```bash
cd ~/tt-ign/tt-metal
mkdir -p models/experimental/hunyuan_image_3_0/ref/tokenizer/assets

hf download tencent/HunyuanImage-3.0 \
  tokenizer.json tokenizer_config.json \
  --local-dir models/experimental/hunyuan_image_3_0/ref/tokenizer/assets
```

If `hf` is not installed, use `huggingface-cli download` with the same arguments.

Verify:

```bash
ls -lh models/experimental/hunyuan_image_3_0/ref/tokenizer/assets/tokenizer.json
```

Load check:

```bash
python3 -c "from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer; HunyuanTokenizer.from_pretrained(); print('OK')"
```

Sanity validation:

```bash
python3 -m models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer
```
