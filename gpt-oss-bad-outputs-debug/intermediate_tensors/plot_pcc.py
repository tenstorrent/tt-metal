import torch
from models.common.utility_functions import comp_pcc

users_to_plot = [14, 46, 78]

pre_embed = {
    user_id: torch.load(f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_embed_user_id{user_id}.pt")
    for user_id in users_to_plot
}

passed, pcc = comp_pcc(pre_embed[users_to_plot[0]], pre_embed[users_to_plot[1]])
print(f"Prefill pre embed user {users_to_plot[0]} vs user {users_to_plot[1]}: {passed}, {pcc}")
passed, pcc = comp_pcc(pre_embed[users_to_plot[0]], pre_embed[users_to_plot[2]])
print(f"Prefill pre embed user {users_to_plot[0]} vs user {users_to_plot[2]}: {passed}, {pcc}")
passed, pcc = comp_pcc(pre_embed[users_to_plot[1]], pre_embed[users_to_plot[2]])
print(f"Prefill pre embed user {users_to_plot[1]} vs user {users_to_plot[2]}: {passed}, {pcc}")

post_embed = {
    user_id: torch.load(f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_embed_user_id{user_id}.pt")
    for user_id in users_to_plot
}

passed, pcc = comp_pcc(post_embed[users_to_plot[0]], post_embed[users_to_plot[1]])
print(f"Prefill post embed user {users_to_plot[0]} vs user {users_to_plot[1]}: {passed}, {pcc}")
passed, pcc = comp_pcc(post_embed[users_to_plot[0]], post_embed[users_to_plot[2]])
print(f"Prefill post embed user {users_to_plot[0]} vs user {users_to_plot[2]}: {passed}, {pcc}")
passed, pcc = comp_pcc(post_embed[users_to_plot[1]], post_embed[users_to_plot[2]])
print(f"Prefill post embed user {users_to_plot[1]} vs user {users_to_plot[2]}: {passed}, {pcc}")

pre_rmsnorm = {
    user_id: torch.load(f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_rmsnorm_user_id{user_id}.pt")
    for user_id in users_to_plot
}

passed, pcc = comp_pcc(pre_rmsnorm[users_to_plot[0]], pre_rmsnorm[users_to_plot[1]])
print(f"Prefill pre rmsnorm user {users_to_plot[0]} vs user {users_to_plot[1]}: {passed}, {pcc}")
passed, pcc = comp_pcc(pre_rmsnorm[users_to_plot[0]], pre_rmsnorm[users_to_plot[2]])
print(f"Prefill pre rmsnorm user {users_to_plot[0]} vs user {users_to_plot[2]}: {passed}, {pcc}")
passed, pcc = comp_pcc(pre_rmsnorm[users_to_plot[1]], pre_rmsnorm[users_to_plot[2]])
print(f"Prefill pre rmsnorm user {users_to_plot[1]} vs user {users_to_plot[2]}: {passed}, {pcc}")

pre_attn = {
    user_id: torch.load(f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_attn_user_id{user_id}.pt")
    for user_id in users_to_plot
}

passed, pcc = comp_pcc(pre_attn[users_to_plot[0]], pre_attn[users_to_plot[1]])
print(f"Prefill pre attn user {users_to_plot[0]} vs user {users_to_plot[1]}: {passed}, {pcc}")
passed, pcc = comp_pcc(pre_attn[users_to_plot[0]], pre_attn[users_to_plot[2]])
print(f"Prefill pre attn user {users_to_plot[0]} vs user {users_to_plot[2]}: {passed}, {pcc}")
passed, pcc = comp_pcc(pre_attn[users_to_plot[1]], pre_attn[users_to_plot[2]])
print(f"Prefill pre attn user {users_to_plot[1]} vs user {users_to_plot[2]}: {passed}, {pcc}")

post_attn = {
    user_id: torch.load(f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_attn_user_id{user_id}.pt")
    for user_id in users_to_plot
}

passed, pcc = comp_pcc(post_attn[users_to_plot[0]], post_attn[users_to_plot[1]])
print(f"Prefill post attn user {users_to_plot[0]} vs user {users_to_plot[1]}: {passed}, {pcc}")
passed, pcc = comp_pcc(post_attn[users_to_plot[0]], post_attn[users_to_plot[2]])
print(f"Prefill post attn user {users_to_plot[0]} vs user {users_to_plot[2]}: {passed}, {pcc}")
passed, pcc = comp_pcc(post_attn[users_to_plot[1]], post_attn[users_to_plot[2]])
print(f"Prefill post attn user {users_to_plot[1]} vs user {users_to_plot[2]}: {passed}, {pcc}")

post_mlp = {
    user_id: torch.load(f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_mlp_user_id{user_id}.pt")
    for user_id in users_to_plot
}
passed, pcc = comp_pcc(post_mlp[users_to_plot[0]], post_mlp[users_to_plot[1]])
print(f"Prefill post mlp user {users_to_plot[0]} vs user {users_to_plot[1]}: {passed}, {pcc}")
passed, pcc = comp_pcc(post_mlp[users_to_plot[0]], post_mlp[users_to_plot[2]])
print(f"Prefill post mlp user {users_to_plot[0]} vs user {users_to_plot[2]}: {passed}, {pcc}")
passed, pcc = comp_pcc(post_mlp[users_to_plot[1]], post_mlp[users_to_plot[2]])
print(f"Prefill post mlp user {users_to_plot[1]} vs user {users_to_plot[2]}: {passed}, {pcc}")


# prefill_user0 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_embed_user_id0.pt")
# prefill_user32 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_embed_user_id32.pt")
# prefill_user64 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_embed_user_id64.pt")

# passed, pcc = comp_pcc(prefill_user0, prefill_user32)
# print(f"Prefill post embed user 0 vs user 32: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user0, prefill_user64)
# print(f"Prefill post embed user 0 vs user 64: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user32, prefill_user64)
# print(f"Prefill post embed user 32 vs user 64: {passed}, {pcc}")


# prefill_user0 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_rmsnorm_user_id0.pt")
# prefill_user32 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_rmsnorm_user_id32.pt")
# prefill_user64 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_rmsnorm_user_id64.pt")

# passed, pcc = comp_pcc(prefill_user0, prefill_user32)
# print(f"Prefill pre rmsnorm user 0 vs user 32: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user0, prefill_user64)
# print(f"Prefill pre rmsnorm user 0 vs user 64: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user32, prefill_user64)
# print(f"Prefill pre rmsnorm user 32 vs user 64: {passed}, {pcc}")


# prefill_user0 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_attn_user_id0.pt")
# prefill_user32 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_attn_user_id32.pt")
# prefill_user64 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_attn_user_id64.pt")

# passed, pcc = comp_pcc(prefill_user0, prefill_user32)
# print(f"Prefill pre attn user 0 vs user 32: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user0, prefill_user64)
# print(f"Prefill pre attn user 0 vs user 64: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user32, prefill_user64)
# print(f"Prefill pre attn user 32 vs user 64: {passed}, {pcc}")


# prefill_user0 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_attn_user_id0.pt")
# prefill_user32 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_attn_user_id32.pt")
# prefill_user64 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_attn_user_id64.pt")

# passed, pcc = comp_pcc(prefill_user0, prefill_user32)
# print(f"Prefill post attn user 0 vs user 32: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user0, prefill_user64)
# print(f"Prefill post attn user 0 vs user 64: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user32, prefill_user64)
# print(f"Prefill post attn user 32 vs user 64: {passed}, {pcc}")

# prefill_user0 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_mlp_user_id0.pt")
# prefill_user32 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_mlp_user_id32.pt")
# prefill_user64 = torch.load("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_mlp_user_id64.pt")

# passed, pcc = comp_pcc(prefill_user0, prefill_user32)
# print(f"Prefill post mlp user 0 vs user 32: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user0, prefill_user64)
# print(f"Prefill post mlp user 0 vs user 64: {passed}, {pcc}")
# passed, pcc = comp_pcc(prefill_user32, prefill_user64)
# print(f"Prefill post mlp user 32 vs user 64: {passed}, {pcc}")

decode_comp_pcc_0 = []
decode_comp_pcc_1 = []
decode_comp_pcc_2 = []
for i in range(74, 274):
    decode_out = {
        user_id: torch.load(
            f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_attn_pos{i}_user_id{user_id}.pt"
        )
        for user_id in users_to_plot
    }

    passed, pcc = comp_pcc(decode_out[users_to_plot[0]], decode_out[users_to_plot[1]])
    decode_comp_pcc_0.append(pcc)
    passed, pcc = comp_pcc(decode_out[users_to_plot[0]], decode_out[users_to_plot[2]])
    decode_comp_pcc_1.append(pcc)
    passed, pcc = comp_pcc(decode_out[users_to_plot[1]], decode_out[users_to_plot[2]])
    decode_comp_pcc_2.append(pcc)

import matplotlib.pyplot as plt

plt.plot(decode_comp_pcc_0)
plt.plot(decode_comp_pcc_1)
plt.plot(decode_comp_pcc_2)
plt.legend(
    [
        f"{users_to_plot[0]} vs {users_to_plot[1]}",
        f"{users_to_plot[0]} vs {users_to_plot[2]}",
        f"{users_to_plot[1]} vs {users_to_plot[2]}",
    ]
)
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pcc.png")
