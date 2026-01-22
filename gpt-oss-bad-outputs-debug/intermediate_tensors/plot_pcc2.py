import torch
from models.common.utility_functions import comp_pcc
import matplotlib.pyplot as plt

iter_to_compare = 0
positions_to_compare = [0, 2]

prefill_tt_k0 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_tt_k_user_id{positions_to_compare[0]}.pt"
).squeeze()
prefill_tt_k1 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_tt_k_user_id{positions_to_compare[1]}.pt"
).squeeze()
prefill_tt_v0 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_tt_v_user_id{positions_to_compare[0]}.pt"
).squeeze()
prefill_tt_v1 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_tt_v_user_id{positions_to_compare[1]}.pt"
).squeeze()

prefill_k_cache0 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_k_cache_user_id{positions_to_compare[0]}.pt"
).squeeze()
prefill_k_cache1 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_k_cache_user_id{positions_to_compare[1]}.pt"
).squeeze()
prefill_v_cache0 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_v_cache_user_id{positions_to_compare[0]}.pt"
).squeeze()
prefill_v_cache1 = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_v_cache_user_id{positions_to_compare[1]}.pt"
).squeeze()

decode_pre_rmsnorm = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_rmsnorm_iter{iter_to_compare}.pt"
).squeeze()
decode_pre_attn = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_attn_iter{iter_to_compare}.pt"
).squeeze()
decode_pre_create_qkv_heads = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_create_qkv_heads_iter{iter_to_compare}.pt"
).squeeze()
decode_pre_sdpa = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_sdpa_iter{iter_to_compare}.pt"
).squeeze()
decode_k_cache = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_k_cache_iter{iter_to_compare}.pt"
).squeeze()
decode_v_cache = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_v_cache_iter{iter_to_compare}.pt"
).squeeze()
decode_page_table = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_page_table_iter{iter_to_compare}.pt"
).squeeze()
decode_post_sdpa = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_sdpa_iter{iter_to_compare}.pt"
).squeeze()
decode_post_create_qkv_heads_q = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_create_qkv_heads_q_iter{iter_to_compare}.pt"
).squeeze()
decode_post_create_qkv_heads_k = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_create_qkv_heads_k_iter{iter_to_compare}.pt"
).squeeze()
decode_post_create_qkv_heads_v = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_create_qkv_heads_v_iter{iter_to_compare}.pt"
).squeeze()
decode_post_attn = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_attn_iter{iter_to_compare}.pt"
).squeeze()
decode_post_mlp = torch.load(
    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_mlp_iter{iter_to_compare}.pt"
).squeeze()


plt.plot(decode_pre_rmsnorm[positions_to_compare[0]].to(torch.float))
plt.plot(decode_pre_rmsnorm[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_rmsnorm.png")
passed, pcc = comp_pcc(decode_pre_rmsnorm[positions_to_compare[0]], decode_pre_rmsnorm[positions_to_compare[1]])
print(f"Decode pre rmsnorm: {passed}, {pcc}")

plt.clf()
plt.plot(decode_pre_attn[positions_to_compare[0]].to(torch.float))
plt.plot(decode_pre_attn[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_attn.png")
passed, pcc = comp_pcc(decode_pre_attn[positions_to_compare[0]], decode_pre_attn[positions_to_compare[1]])
print(f"Decode pre attn: {passed}, {pcc}")

plt.clf()
plt.plot(decode_pre_create_qkv_heads[positions_to_compare[0]].to(torch.float))
plt.plot(decode_pre_create_qkv_heads[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_create_qkv_heads.png")
passed, pcc = comp_pcc(
    decode_pre_create_qkv_heads[positions_to_compare[0]], decode_pre_create_qkv_heads[positions_to_compare[1]]
)
print(f"Decode pre create qkv heads: {passed}, {pcc}")

plt.clf()
plt.plot(decode_post_create_qkv_heads_q[positions_to_compare[0]].to(torch.float))
plt.plot(decode_post_create_qkv_heads_q[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_create_qkv_heads_q.png")
passed, pcc = comp_pcc(
    decode_post_create_qkv_heads_q[positions_to_compare[0]], decode_post_create_qkv_heads_q[positions_to_compare[1]]
)
print(f"Decode post create qkv heads q: {passed}, {pcc}")

plt.clf()
plt.plot(decode_post_create_qkv_heads_k[positions_to_compare[0]].to(torch.float))
plt.plot(decode_post_create_qkv_heads_k[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_create_qkv_heads_k.png")
passed, pcc = comp_pcc(
    decode_post_create_qkv_heads_k[positions_to_compare[0]], decode_post_create_qkv_heads_k[positions_to_compare[1]]
)
print(f"Decode post create qkv heads k: {passed}, {pcc}")

plt.clf()
plt.plot(decode_post_create_qkv_heads_v[positions_to_compare[0]].to(torch.float))
plt.plot(decode_post_create_qkv_heads_v[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_create_qkv_heads_v.png")
passed, pcc = comp_pcc(
    decode_post_create_qkv_heads_v[positions_to_compare[0]], decode_post_create_qkv_heads_v[positions_to_compare[1]]
)
print(f"Decode post create qkv heads v: {passed}, {pcc}")

plt.clf()
plt.plot(decode_pre_sdpa[positions_to_compare[0]].to(torch.float))
plt.plot(decode_pre_sdpa[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_sdpa.png")
passed, pcc = comp_pcc(decode_pre_sdpa[positions_to_compare[0]], decode_pre_sdpa[positions_to_compare[1]])
print(f"Decode pre sdpa: {passed}, {pcc}")

plt.clf()
plt.plot(decode_post_sdpa[positions_to_compare[0]].to(torch.float))
plt.plot(decode_post_sdpa[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_sdpa.png")
passed, pcc = comp_pcc(decode_post_sdpa[positions_to_compare[0]], decode_post_sdpa[positions_to_compare[1]])
print(f"Decode post sdpa: {passed}, {pcc}")


kv_cache_blocks_to_compare = 64

plt.clf()
plt.plot(prefill_tt_k0.flatten().to(torch.float))
plt.plot(prefill_tt_k1.flatten().to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_tt_k.png")
passed, pcc = comp_pcc(prefill_tt_k0, prefill_tt_k1)
print(f"Prefill tt k: {passed}, {pcc}")

plt.clf()
plt.plot(prefill_tt_v0.flatten().to(torch.float))
plt.plot(prefill_tt_v1.flatten().to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_tt_v.png")
passed, pcc = comp_pcc(prefill_tt_v0, prefill_tt_v1)
print(f"Prefill tt v: {passed}, {pcc}")

plt.clf()
fig, ax = plt.subplots(3)
ax[0].plot(
    prefill_k_cache0[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
ax[0].plot(
    prefill_k_cache1[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
ax[0].legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
ax[1].plot(
    prefill_k_cache0[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]][:, 0]
    .flatten()
    .to(torch.float)
)
ax[1].plot(
    prefill_k_cache1[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]][:, 0]
    .flatten()
    .to(torch.float)
)
ax[2].plot(
    prefill_k_cache0[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]][:, 0, 0]
    .flatten()
    .to(torch.float)
)
ax[2].plot(
    prefill_k_cache1[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]][:, 0, 0]
    .flatten()
    .to(torch.float)
)
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_k_cache.png")
passed, pcc = comp_pcc(
    prefill_k_cache0[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]],
    prefill_k_cache1[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]],
)
print(f"Prefill k cache: {passed}, {pcc}")

plt.clf()
plt.plot(
    prefill_v_cache0[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
plt.plot(
    prefill_v_cache1[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_v_cache.png")
passed, pcc = comp_pcc(
    prefill_v_cache0[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]],
    prefill_v_cache1[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]],
)
print(f"Prefill v cache: {passed}, {pcc}")

plt.clf()
fig, ax = plt.subplots(3)
ax[0].plot(
    decode_k_cache[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
ax[0].plot(
    decode_k_cache[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
ax[0].legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
ax[1].plot(
    decode_k_cache[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]][:, 0]
    .flatten()
    .to(torch.float)
)
ax[1].plot(
    decode_k_cache[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]][:, 0]
    .flatten()
    .to(torch.float)
)
ax[2].plot(
    decode_k_cache[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]][:, 0, 0]
    .flatten()
    .to(torch.float)
)
ax[2].plot(
    decode_k_cache[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]][:, 0, 0]
    .flatten()
    .to(torch.float)
)
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_k_cache.png")
passed, pcc = comp_pcc(
    decode_k_cache[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]],
    decode_k_cache[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]],
)
print(f"Decode k cache: {passed}, {pcc}")

plt.clf()
plt.plot(
    decode_v_cache[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
plt.plot(
    decode_v_cache[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]].flatten().to(torch.float)
)
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_v_cache.png")
passed, pcc = comp_pcc(
    decode_v_cache[decode_page_table[positions_to_compare[0]][:kv_cache_blocks_to_compare]],
    decode_v_cache[decode_page_table[positions_to_compare[1]][:kv_cache_blocks_to_compare]],
)
print(f"Decode v cache: {passed}, {pcc}")


plt.plot(decode_post_attn[positions_to_compare[0]].to(torch.float))
plt.plot(decode_post_attn[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_attn.png")
passed, pcc = comp_pcc(decode_post_attn[positions_to_compare[0]], decode_post_attn[positions_to_compare[1]])
print(f"Decode post attn: {passed}, {pcc}")

plt.clf()
plt.plot(decode_post_mlp[positions_to_compare[0]].to(torch.float))
plt.plot(decode_post_mlp[positions_to_compare[1]].to(torch.float))
plt.legend([f"User {positions_to_compare[0]}", f"User {positions_to_compare[1]}"])
plt.show()
plt.savefig("gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_mlp.png")
passed, pcc = comp_pcc(decode_post_mlp[positions_to_compare[0]], decode_post_mlp[positions_to_compare[1]])
print(f"Decode post mlp: {passed}, {pcc}")
