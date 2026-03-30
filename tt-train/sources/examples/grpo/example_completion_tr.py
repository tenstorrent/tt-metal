# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from utils.inference_tr import (
    setup_tt_transformers_inference,
    completion_batched_multiple_prompts_tr,
)


def capitals_batched():
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.ETH if ttnn.GetNumAvailableDevices() > 1 else ttnn.DispatchCoreType.WORKER
        ),
    )

    ctx = setup_tt_transformers_inference(
        mesh_device=mesh_device,
        max_seq_len=1024,
        max_batch_size=32,
        max_tokens_to_complete=64,
        temperature=0.0,
        group_size=1,
        instruct=True,
    )

    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom of Great Britain and Northern Ireland is",
        "The capital of Czech Republic is",
        # "The capital of France is",
        # "The capital of Portugal is",
        # "The capital of United Kingdom of Great Britain and Northern Ireland is",
        # "The capital of Czech Republic is",
    ] * 8

    # Tokenize prompts (encode_prompt applies chat template for instruct models)
    prompts_tokenized = [ctx.model_args.encode_prompt(p, instruct=True) for p in user_prompts]

    completions = completion_batched_multiple_prompts_tr(ctx, prompts_tokenized)

    print()
    for prompt_text, completion_tokens in zip(user_prompts, completions):
        answer = ctx.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        print(f"Q: {prompt_text}")
        print(f"A: {answer}")
        print()

    del ctx
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    capitals_batched()
