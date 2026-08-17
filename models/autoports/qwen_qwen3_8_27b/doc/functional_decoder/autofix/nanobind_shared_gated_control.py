"""No-device control importing the pre-existing gated-attention dependency."""

from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _get_paged_sdpa_decode_program_config,
    apply_rotary_pos_emb_ttnn,
)

print(
    "SHARED_GATED_IMPORT_CONTROL_OK "
    f"symbols={_get_paged_sdpa_decode_program_config.__name__},{apply_rotary_pos_emb_ttnn.__name__}"
)
