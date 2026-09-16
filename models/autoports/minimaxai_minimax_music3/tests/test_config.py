from models.autoports.minimaxai_minimax_music3 import config as C


def test_config_from_snapshot(snapshot):
    cfg = C.Music3Config.from_snapshot(snapshot)
    assert cfg.llm.hidden_size == 4096 and cfg.llm.num_hidden_layers == 36 and cfg.llm.vocab_size == 200000
    assert (
        cfg.depth.num_layers == 4
        and cfg.depth.head_dim == 256
        and cfg.dit.inner_dim == 2048
        and cfg.dit.concat_channels == 2304
    )
    assert cfg.vocoder.upsampling_ratios == (8, 8, 4, 2)


def test_latent_length_and_chunks():
    assert C.latent_length_for_frames(200) == 689 and C.latent_length_for_frames(100) == 344
    assert (
        C.chunk_starts(200) == [0]
        and C.chunk_starts(300) == [0, 100]
        and C.chunk_starts(1500) == list(range(0, 1400, 100))
    )
    assert C.SLICED_VOCAB == 16385 and C.SLICED_VOCAB_PADDED % 32 == 0
