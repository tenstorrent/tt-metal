import torch 
import ttnn
from loguru import logger
import numpy as np
from models.experimental.functional_deepseek_llm.tt.preprocessor import custom_preprocessor
from models.experimental.functional_deepseek_llm.demo.demo_utils import gen_position_ids, generate
from models.generation_utils import get_logits_processor 
from transformers import AutoTokenizer, AutoConfig

# To be tested due to RAM shortage currently

def test_demo():
    
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    config = AutoConfig.from_pretrained("/home/mcw/saiprasad/deepseek-llm/test/saved_model")
    state_dict = torch.load("/home/mcw/saiprasad/deepseek-llm/test/model_weights.pth")
    parameters = custom_preprocessor(device, state_dict)

    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = gen_position_ids(input_ids)
    
    logits_processor = get_logits_processor(input_ids, config)

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers
    
    generated_ids = generate(
        config, input_ids, attention_mask, position_ids, parameters, hidden_size, num_attention_heads, num_hidden_layers, logits_processor, tokenizer, device
    )
    
    tt_generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    logger.info(f"output : {tt_generated_text}")
    
    ttnn.close_device(device)
    
if __name__ == "__main__":
    test_demo()