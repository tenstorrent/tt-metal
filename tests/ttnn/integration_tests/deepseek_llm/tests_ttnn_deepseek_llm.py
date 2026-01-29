import torch
import ttnn
import pytest
from loguru import logger
import numpy as np
from models.experimental.functional_deepseek_llm.tt.preprocessor import custom_preprocessor
from models.utility_functions import comp_pcc
from models.experimental.functional_deepseek_llm.demo.demo_utils import gen_position_ids
from models.experimental.functional_deepseek_llm.tt.ttnn_deepseek_llm import TT_LlamaRMSNorm, TT_LlamaMLP, TT_LlamaRotaryEmbedding, TT_LlamaAttention, TT_LlamaDecoderLayer, TT_LlamaModel, TT_LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

@pytest.mark.parametrize("model_name", ["deepseek-ai/deepseek-llm-7b-chat"])
def test_base_RMSNorm(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    config = model.config
    
    hidden_states = torch.load("hidden_states.pt")

    for i in range(config.num_hidden_layers):
        base_RMSNorm = model.model.layers[i].post_attention_layernorm
        
        with torch.no_grad():

            base_RMSNorm_output = base_RMSNorm(
                hidden_states, 
            )[0]

    torch.save(base_RMSNorm_output, "base_RMSNorm_output.pt") 

def test_tt_RMSNorm(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)

    config = AutoConfig.from_pretrained("saved_model")
    state_dict = torch.load("model_weights.pth")

    hidden_states = torch.load("hidden_states.pt")
    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)

    parameters = custom_preprocessor(device, state_dict)

    for i in range(config.num_hidden_layers):
        
        path = f"model.layers.{i}"
        layer_name = "post_attention_layernorm"
        
        ttnn_RMSNorm= TT_LlamaRMSNorm(parameters, path, layer_name)  
        
        with torch.no_grad():

            ttnn_RMSNorm_output = ttnn_RMSNorm(
                ttnn_hidden_states, 
            )
    
    ttnn_RMSNorm_output = ttnn.to_torch(ttnn_RMSNorm_output)
    base_RMSNorm_output = torch.load("base_RMSNorm_output.pt")

    pcc_message, pcc_score = comp_pcc(base_RMSNorm_output, ttnn_RMSNorm_output)
    
    logger.info(f"TT RMSNorm PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)
    
def test_base_MLP(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    config = model.config

    hidden_states = torch.load("hidden_states.pt")
    for i in range(config.num_hidden_layers):
        base_MLP = model.model.layers[i].mlp
        
        with torch.no_grad():
            base_MLP_output = base_MLP(
                hidden_states, 
            )

    torch.save(base_MLP_output, "base_MLP_output.pt") 
    
def test_tt_MLP(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)

    config = AutoConfig.from_pretrained("saved_model")
    state_dict = torch.load("model_weights.pth")

    hidden_states = torch.load("hidden_states.pt")
    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)

    parameters = custom_preprocessor(device, state_dict)

    for i in range(config.num_hidden_layers):
            
        path = f"model.layers.{i}"
        
        ttnn_MLP= TT_LlamaMLP(parameters, path)  
        
        with torch.no_grad():

            ttnn_MLP_output = ttnn_MLP(
                ttnn_hidden_states, 
                device
            )
            
    ttnn_MLP_output = ttnn.to_torch(ttnn_MLP_output)
    base_MLP_output = torch.load("base_MLP_output.pt")

    pcc_message, pcc_score = comp_pcc(base_MLP_output, ttnn_MLP_output)
    
    logger.info(f"TT MLP PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)
    
def test_base_RotaryEmb(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    position_ids = gen_position_ids(input_ids)
    hidden_states = torch.load("hidden_states.pt")
    
    base_rotary_emb = model.model.rotary_emb
    
    with torch.no_grad():

        base_rotary_emb_output = base_rotary_emb(
            hidden_states, 
            position_ids
        )[0]

    torch.save(base_rotary_emb_output, "base_rotary_emb_output.pt") 
    
def test_TTRotaryEmbedding(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)
    config = AutoConfig.from_pretrained("saved_model")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    position_ids = gen_position_ids(input_ids)

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    
    ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)

    ttnn_rotary_emb= TT_LlamaRotaryEmbedding(hidden_size, num_heads, device)  
    
    with torch.no_grad():

        ttnn_rotary_emb_output = ttnn_rotary_emb(
            ttnn_position_ids, 
        )[0]
        
    ttnn_rotary_emb_output = ttnn.to_torch(ttnn_rotary_emb_output)
    base_rotary_emb_output = torch.load("base_rotary_emb_output.pt")
    
    pcc_message, pcc_score = comp_pcc(base_rotary_emb_output, ttnn_rotary_emb_output)
    
    logger.info(f"TT Rotary Embedding PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)

def test_base_Attention(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    config = model.config

    hidden_states = torch.load("hidden_states.pt")
    position_embeddings = torch.load("position_embeddings.pt")
    attention_mask = None
    
    for i in range(config.num_hidden_layers):
        base_attention = model.model.layers[i].self_attn
        
        with torch.no_grad():
            base_attention_output = base_attention(
                hidden_states, 
                position_embeddings, 
                attention_mask
            )[0]

    torch.save(base_attention_output, "base_attention_output.pt") 
    
def test_TTAttention(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)
    
    config = AutoConfig.from_pretrained("saved_model")
    state_dict = torch.load("model_weights.pth")
    
    hidden_states = torch.load("hidden_states.pt")
    position_embeddings = torch.load("position_embeddings.pt")
    attention_mask = None
    
    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    
    parameters = custom_preprocessor(device, state_dict)
    
    tt_position_embeddings_1 = ttnn.from_torch(position_embeddings[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    tt_position_embeddings_2 = ttnn.from_torch(position_embeddings[1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    
    tt_position_embeddings = (tt_position_embeddings_1, tt_position_embeddings_2)
    
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads

    for i in range(config.num_hidden_layers):
            
        path = f"model.layers.{i}"
        
        ttnn_Attention= TT_LlamaAttention(parameters, path, hidden_size, num_attention_heads)  
        
        with torch.no_grad():

            ttnn_attention_output = ttnn_Attention(
                ttnn_hidden_states, 
                position_embeddings = tt_position_embeddings,
                attention_mask = attention_mask,
                device = device
            )[0]
                
    base_attention_output = torch.load("base_attention_output.pt")
    ttnn_attention_output = ttnn.to_torch(ttnn_attention_output)
    
    pcc_message, pcc_score = comp_pcc(base_attention_output, ttnn_attention_output)
    
    logger.info(f"TT Attention PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)
    
def test_base_decoder(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    config = model.config
    inputs = torch.load("input_tensors.pt")
    
    hidden_states = inputs['hidden_states']
    position_embeddings = inputs["position_embeddings"]
    
    for i in range(config.num_hidden_layers):
        base_decoder = model.model.layers[i]
        
        with torch.no_grad():
            base_decoder_output = base_decoder(
                hidden_states = hidden_states,
                position_embeddings = position_embeddings
            )[0]
            

    torch.save(base_decoder_output, "base_decoder_output.pt") 
    
def test_TTdecoder(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)
    
    config = AutoConfig.from_pretrained("saved_model")
    state_dict = torch.load("model_weights.pth")
    
    inputs = torch.load("input_tensors.pt")
    
    hidden_states = inputs['hidden_states']
    position_embeddings = inputs["position_embeddings"]

    ttnn_hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    
    parameters = custom_preprocessor(device, state_dict)
    
    tt_position_embeddings_1 = ttnn.from_torch(position_embeddings[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    tt_position_embeddings_2 = ttnn.from_torch(position_embeddings[1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    
    tt_position_embeddings = (tt_position_embeddings_1, tt_position_embeddings_2)

    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads

    for i in range(config.num_hidden_layers):
        
        path = f"model.layers.{i}"
        
        tt_decoder = TT_LlamaDecoderLayer(parameters, path, hidden_size, num_attention_heads)  

        with torch.no_grad():

            tt_decoder_output = tt_decoder(
                hidden_states = ttnn_hidden_states,
                position_embeddings = tt_position_embeddings,
                device=device
            )[0]  
            
    tt_decoder_output = ttnn.to_torch(tt_decoder_output)
    base_decoder_output = torch.load("base_decoder_output.pt")
    
    pcc_message, pcc_score = comp_pcc(base_decoder_output, tt_decoder_output)

    logger.info(f"TT Decoder layer PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)

def test_base_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = gen_position_ids(input_ids)

    base_model = model.model

    with torch.no_grad():
        
        base_model_output = base_model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )

    torch.save(base_model_output, "base_model_output.pt")

    
def test_TT_model(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    config = AutoConfig.from_pretrained("saved_model")
    state_dict = torch.load("model_weights.pth")
    parameters = custom_preprocessor(device, state_dict)
    
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    attention_mask = None
    position_ids = gen_position_ids(input_ids)

    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device = device)
    
    ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
        
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers
    
    tt_model = TT_LlamaModel(config, parameters, hidden_size, num_attention_heads, num_hidden_layers, device)  
    
    with torch.no_grad():

        tt_model_output = tt_model(
            ttnn_input_ids, attention_mask=attention_mask, position_ids=ttnn_position_ids, device = device
        )
    
    base_model_output = torch.load("base_model_output.pt")
    tt_model_output_2 = ttnn.to_torch(tt_model_output)

    pcc_message, pcc_score = comp_pcc(base_model_output, tt_model_output_2)

    logger.info(f"TT Model PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)

def test_base_causalLMmodel(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = gen_position_ids(input_ids)
    
    base_model = model

    with torch.no_grad():
        
        base_model_output = base_model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )[0]

    torch.save(base_model_output, "base_CasualLMmodel_output.pt") 

def test_TT_causalLMmodel(model_name):
    device = ttnn.open_device(device_id = 0)
    ttnn.SetDefaultDevice(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained("saved_model")
    state_dict = torch.load("model_weights.pth")
    parameters = custom_preprocessor(device, state_dict)
    
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs.input_ids
    attention_mask = None
    position_ids = gen_position_ids(input_ids)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device = device)
    
    ttnn_position_ids = ttnn.from_torch(position_ids, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device = device)
    
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers
    
    tt_CausalLMmodel = TT_LlamaForCausalLM(config, parameters, hidden_size, num_attention_heads, num_hidden_layers, device)  
    
    with torch.no_grad():

        tt_CausalLMmodel_output = tt_CausalLMmodel(
            ttnn_input_ids, attention_mask=attention_mask, position_ids=ttnn_position_ids, device = device
        )[1]
        
    tt_CausalLMmodel_output = ttnn.to_torch(tt_CausalLMmodel_output)
    
    base_CasualLMmodel_output = torch.load("base_CasualLMmodel_output.pt")

    pcc_message, pcc_score = comp_pcc(base_CasualLMmodel_output, tt_CausalLMmodel_output)

    logger.info(f"TT CausalLM Model PCC score: {pcc_score} \n")
    
    ttnn.close_device(device)