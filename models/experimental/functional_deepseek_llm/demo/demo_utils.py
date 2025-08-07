import torch
import ttnn
from models.experimental.functional_deepseek_llm.tt.ttnn_deepseek_llm import TT_LlamaForCausalLM

def gen_position_ids(input_ids):
    
    past_key_values_length = 0
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=None,
    )

    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids

def generate(
    config, input_ids, attention_mask, position_ids, parameters, hidden_size, num_attention_heads, num_hidden_layers, logits_processor, tokenizer,device, num_tokens = 2
):    
    
    tt_CausalLMmodel = TT_LlamaForCausalLM(config, parameters, hidden_size, num_attention_heads, num_hidden_layers, device)

    for i in range(num_tokens):
        
        ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device = device)
    
        ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        
        tt_out = tt_CausalLMmodel(ttnn_input_ids, attention_mask=attention_mask, position_ids=ttnn_position_ids, device = device)[1]
        
        tt_out = ttnn.to_torch(tt_out)
        
        next_token_logits = tt_out[:, -1, :]
        
        processed_logits = logits_processor(input_ids, next_token_logits)
        next_token = torch.argmax(processed_logits, dim=-1)
        
        s = tokenizer.decode(next_token.item(), skip_special_tokens=True)
        print(f"next word : {s}") 
        
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
        
        position_ids = gen_position_ids(input_ids)
    
    return input_ids