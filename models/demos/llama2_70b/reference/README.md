# Reference implementation from Llama github

```bash
pip install -r requirements.txt

python example_text_completion.py \
    --ckpt_dir /proj_sw/user_dev/llama-data/llama-2-70b/ \
    --tokenizer_path /proj_sw/user_dev/llama-data/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4 --skip-model-load

python example_chat_completion.py \
    --ckpt_dir /proj_sw/user_dev/llama-data/llama-2-7b/llama-2-7b \
    --tokenizer_path /proj_sw/user_dev/llama-data/tokenizer.model \
    --max_seq_len 128 --max_batch_size 1 --skip-model-load
```
