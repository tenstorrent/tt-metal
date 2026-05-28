import torch
def convert_phi_weights(hf_state_dict, config):
    tt_state_dict = {"embd.weight": hf_state_dict["model.embed_tokens.weight"]}
    for i in range(config.num_hidden_layers):
        tt_state_dict[f"layers.{i}.ln.weight"] = hf_state_dict[f"model.layers.{i}.input_layernorm.weight"]
        tt_state_dict[f"layers.{i}.ln.bias"] = hf_state_dict[f"model.layers.{i}.input_layernorm.bias"]
        q, k, v = hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"], hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"], hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        tt_state_dict[f"layers.{i}.mixer.Wqkv.weight"] = torch.cat([q, k, v], dim=0)
        qb, kb, vb = hf_state_dict[f"model.layers.{i}.self_attn.q_proj.bias"], hf_state_dict[f"model.layers.{i}.self_attn.k_proj.bias"], hf_state_dict[f"model.layers.{i}.self_attn.v_proj.bias"]
        tt_state_dict[f"layers.{i}.mixer.Wqkv.bias"] = torch.cat([qb, kb, vb], dim=0)
        tt_state_dict[f"layers.{i}.mixer.out_proj.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.out_proj.weight"]
        tt_state_dict[f"layers.{i}.mixer.out_proj.bias"] = hf_state_dict[f"model.layers.{i}.self_attn.out_proj.bias"]
        tt_state_dict[f"layers.{i}.mlp.fc1.weight"] = hf_state_dict[f"model.layers.{i}.mlp.fc1.weight"]
        tt_state_dict[f"layers.{i}.mlp.fc1.bias"] = hf_state_dict[f"model.layers.{i}.mlp.fc1.bias"]
        tt_state_dict[f"layers.{i}.mlp.fc2.weight"] = hf_state_dict[f"model.layers.{i}.mlp.fc2.weight"]
        tt_state_dict[f"layers.{i}.mlp.fc2.bias"] = hf_state_dict[f"model.layers.{i}.mlp.fc2.bias"]
    tt_state_dict["final_norm.weight"], tt_state_dict["final_norm.bias"] = hf_state_dict["model.final_layernorm.weight"], hf_state_dict["model.final_layernorm.bias"]
    tt_state_dict["lm_head.weight"], tt_state_dict["lm_head.bias"] = hf_state_dict["lm_head.weight"], hf_state_dict["lm_head.bias"]
    return tt_state_dict
