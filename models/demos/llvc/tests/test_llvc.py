    torch_prenet = LLVCPrenet(in_channels=1, out_channels=512, num_blocks=12).eval()
    input_torch = torch.randn(batch_size, 1, seq_len)
    with torch.no_grad():
        expected = torch_prenet(input_torch)
    parameters = ttnn.model_preprocessing.preprocess_model_parameters(
        initialize_model=lambda: torch_prenet, device=device,
    )
    ttnn_prenet = TtLLVCPrenet(device, in_channels=1, out_channels=512,
                                num_blocks=12, parameters=parameters)
    input_nhwc = input_torch.permute(0, 2, 1).unsqueeze(1)
    ttnn_input = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output, _ = ttnn_prenet(ttnn_input, batch_size, seq_len)
    actual = ttnn.to_torch(ttnn_output).squeeze(1).permute(0, 2, 1)
    passing, msg = comp_pcc(expected, actual, pcc=0.99)
    print(f"Prenet PCC: {msg}")
    assert passing, f"Prenet PCC failed: {msg}"
@pytest.mark.parametrize("batch_size", [1])
def test_llvc_encoder(device, use_program_cache, batch_size):
    """Unit test for Encoder submodule."""
    torch.manual_seed(42)
    from models.demos.llvc.tt.llvc_model import TtLLVCEncoder
    channels = 512
    seq_len = 320
    dilations = [1, 2, 4, 8, 16, 32, 1, 2]
    torch_encoder = LLVCEncoder(channels, dilations).eval()
    input_torch = torch.randn(batch_size, channels, seq_len)
    with torch.no_grad():
        expected = torch_encoder(input_torch)
    parameters = ttnn.model_preprocessing.preprocess_model_parameters(
        initialize_model=lambda: torch_encoder, device=device,
    )
    ttnn_encoder = TtLLVCEncoder(device, channels=channels,
                                  dilations=dilations, parameters=parameters)
    input_nhwc = input_torch.permute(0, 2, 1).unsqueeze(1)
    ttnn_input = ttnn.from_torch(input_nhwc, dtype=ttnn.bfloat16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_output, _ = ttnn_encoder(ttnn_input, batch_size, seq_len)
    actual = ttnn.to_torch(ttnn_output).squeeze(1).permute(0, 2, 1)
    passing, msg = comp_pcc(expected, actual, pcc=0.99)
    print(f"Encoder PCC: {msg}")
    assert passing, f"Encoder PCC failed: {msg}"
