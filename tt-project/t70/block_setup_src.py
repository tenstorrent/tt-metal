def _build_block_trace_setup(
    *, mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, F, H, W, has_audio, checkpoint_variant="fast"
):
    """Build an ``LTXTransformerBlock`` + device ``forward`` kwargs for the block trace-perf test.

    Mirrors the non-PCC setup of ``test_ltx_transformer_block``: AV loads the 22B checkpoint
    (``strict=False``) and builds the full video+audio+cross-modal device tensors; video harvests
    random scaled weights from a throwaway diffusers block. Returns
    ``(tt_block, forward_kwargs, video_N_real, audio_N_real)``. Perf-only; no PCC.
    """
    checkpoint_22b = _resolve_checkpoint_22b(checkpoint_variant)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    # Real latent grid → SP-padded sequence (mirrors LTXPipeline)
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)
    audio_N, audio_N_real = _audio_seq_lens(F, sp_factor)
    assert video_N % (32 * sp_factor) == 0, f"video_N={video_N} not sp/tile-aligned for sp={sp_factor}"
    assert audio_N % (32 * sp_factor) == 0, f"audio_N={audio_N} not sp/tile-aligned for sp={sp_factor}"

    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_block = _make_tt_block(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
    )

    if has_audio:
        sd = _load_22b_state_dict(num_layers=1, checkpoint_path=checkpoint_22b)
        if sd is None:
            pytest.skip(f"22B checkpoint not found at {checkpoint_22b}")
        block_sd = {
            k[len("transformer_blocks.0.") :]: v for k, v in sd.items() if k.startswith("transformer_blocks.0.")
        }
        tt_block.load_torch_state_dict(block_sd, strict=False)
    else:
        # video mode — harvest a shape-matching state dict from a throwaway block (perf-only).
        dummy = _make_diffusers_video_block()
        _scale_init_(dummy)
        conv = _convert_diffusers_video_block_to_tt(dummy.state_dict(), num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        tt_block.load_torch_state_dict({k: v.detach().clone() for k, v in conv.items()})
        del dummy

    # Inputs (real-grid token count; the TT side gets a zero-padded copy below).
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(1, video_N_real, DIM, dtype=torch.float32)
    context = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    temb = torch.randn(1, 1, 9 * DIM, dtype=torch.float32)  # 9 adaln params
    prompt_temb = torch.randn(1, 1, 2 * DIM, dtype=torch.float32)  # 2 adaln params for prompt

    # TT video tensors
    spatial = _pad_seq_dim(x, video_N, dim=1).unsqueeze(0)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)
    tt_temb = bf16_tensor(
        temb.reshape(9, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
    )
    tt_prompt_temb = bf16_tensor(prompt_temb.reshape(2, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device)
    tt_cos, tt_sin = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    forward_kwargs = dict(
        video_1BND=tt_spatial,
        video_prompt=tt_prompt,
        video_temb=tt_temb,
        video_N=video_N_real,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        video_prompt_temb=tt_prompt_temb,
    )
    if has_audio:
        # Real tokens in [:audio_N_real], zeros in the padded tail (matches the padded audio latent).
        a_x = torch.zeros(1, audio_N, AUDIO_DIM, dtype=torch.float32)
        a_x[:, :audio_N_real, :] = torch.randn(1, audio_N_real, AUDIO_DIM, dtype=torch.float32)
        a_ctx = torch.randn(1, PROMPT_LEN, AUDIO_CTX_DIM, dtype=torch.float32)
        a_temb = torch.randn(1, 1, 9 * AUDIO_DIM, dtype=torch.float32)
        a_prompt_temb = torch.randn(1, 1, 2 * AUDIO_DIM, dtype=torch.float32)
        av_ca_v = torch.randn(1, 1, 5 * DIM, dtype=torch.float32)  # 4 scale-shift + 1 gate
        av_ca_a = torch.randn(1, 1, 5 * AUDIO_DIM, dtype=torch.float32)

        a_cos, a_sin = _tt_rope(_audio_rope_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        vx_cos, vx_sin = _tt_rope(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
        )
        ax_cos, ax_sin = _tt_rope(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        ax_cos_full, ax_sin_full = _tt_rope_full(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, tp_axis=tp_axis
        )

        # Padding masks, same construction as the fast pipeline (audio padded, video aligned).
        a_attn_mask, a_pad_sp, a_pad_full = build_audio_masks(
            audio_N, audio_N_real, mesh_device=mesh_device, sp_axis=sp_axis
        )
        v_pad_sp = build_video_pad_mask(video_N, video_N_real, mesh_device=mesh_device, sp_axis=sp_axis)

        forward_kwargs.update(
            audio_1BND=bf16_tensor_2dshard(
                a_x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
            ),
            audio_prompt=bf16_tensor(a_ctx.unsqueeze(0), device=mesh_device),
            audio_temb=bf16_tensor(
                a_temb.reshape(9, AUDIO_DIM).unsqueeze(1).unsqueeze(1),
                device=mesh_device,
                mesh_axis=tp_axis,
                shard_dim=3,
            ),
            audio_prompt_temb=bf16_tensor(
                a_prompt_temb.reshape(2, AUDIO_DIM).unsqueeze(1).unsqueeze(1), device=mesh_device
            ),
            av_ca_temb=bf16_tensor(
                av_ca_v.reshape(5, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            av_ca_audio_temb=bf16_tensor(
                av_ca_a.reshape(5, AUDIO_DIM).unsqueeze(1).unsqueeze(1),
                device=mesh_device,
                mesh_axis=tp_axis,
                shard_dim=3,
            ),
            audio_N=audio_N,
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            video_cross_pe_cos=vx_cos,
            video_cross_pe_sin=vx_sin,
            audio_cross_pe_cos=ax_cos,
            audio_cross_pe_sin=ax_sin,
            audio_cross_pe_cos_full=ax_cos_full,
            audio_cross_pe_sin_full=ax_sin_full,
            audio_attn_mask=a_attn_mask,
            audio_padding_mask=a_pad_sp,
            audio_padding_mask_full=a_pad_full,
            video_padding_mask=v_pad_sp,
        )
    return tt_block, forward_kwargs, video_N_real, audio_N_real
