export max_seq_len=131072

echo "INPUT 128"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json" --max_seq_len $max_seq_len
echo "INPUT 256"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_256.json" --max_seq_len $max_seq_len
echo "INPUT 1k"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_long_1k.json" --max_seq_len $max_seq_len
echo "INPUT 2k"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_long_2k.json" --max_seq_len $max_seq_len
echo "INPUT 4k"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json" --max_seq_len $max_seq_len
echo "INPUT 8k"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_long_8k.json" --max_seq_len $max_seq_len
echo "INPUT 16k"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_long_16k.json" --max_seq_len $max_seq_len
echo "INPUT 32k"
LLAMA_DIR=/proj_sw/user_dev/hf_data/Llama3/Llama3.1-8B-Instruct pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-batch-1" --input_prompts "models/tt_transformers/demo/sample_prompts/input_data_long_32k.json" --max_seq_len $max_seq_len
