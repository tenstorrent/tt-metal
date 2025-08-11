# Installing **gpt-oss-20b**

## 1. Downloading the weights

```bash
huggingface-cli download unsloth/gpt-oss-20b-BF16 \
    --local-dir /proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16
```

## 2. Running demo.py

```bash
export GPT_OSS_WEIGHTS_PATH=/proj_sw/user_dev/gpt-oss/gpt-oss-20b-BF16
python models/demos/gpt_oss/reference/demo.py
```
