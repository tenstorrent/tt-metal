import torch
from load_checkpoints import permute, reverse_permute


def test_permute_roundtrip(head_dim=128, n_heads=8, n_kv_heads=2):
    # Test Q weights (n_heads)
    q_weight = torch.randn(head_dim * n_heads, 512)  # Random Q weights
    q_bias = torch.randn(head_dim * n_heads)  # Random Q bias

    # Test permute -> inverse_permute roundtrip for Q weight
    n_heads = q_weight.shape[0] // head_dim
    q_permuted = permute(q_weight, n_heads, q_weight.shape[0], q_weight.shape[1])
    q_restored = reverse_permute(q_permuted, n_heads, q_weight.shape[0], q_weight.shape[1])

    # Test permute -> inverse_permute roundtrip for Q bias
    q_bias_permuted = permute(q_bias.unsqueeze(-1), n_heads, q_bias.shape[0], 1)
    q_bias_restored = reverse_permute(q_bias_permuted, n_heads, q_bias.shape[0], 1).squeeze(-1)

    print("Q weight roundtrip test:")
    print(f"Max difference in weights: {(q_weight - q_restored).abs().max().item():.2e}")
    print(f"Max difference in biases: {(q_bias - q_bias_restored).abs().max().item():.2e}")

    # Test K weights and biases (n_kv_heads)
    k_weight = torch.randn(head_dim * n_kv_heads, 512)  # Random K weights
    k_bias = torch.randn(head_dim * n_kv_heads)  # Random K bias

    # Test permute -> inverse_permute roundtrip for K weight
    k_permuted = permute(k_weight, n_kv_heads, k_weight.shape[0], k_weight.shape[1])
    k_restored = reverse_permute(k_permuted, n_kv_heads, k_weight.shape[0], k_weight.shape[1])

    # Test permute -> inverse_permute roundtrip for K bias
    k_bias_permuted = permute(k_bias.unsqueeze(-1), n_kv_heads, k_bias.shape[0], 1)
    k_bias_restored = reverse_permute(k_bias_permuted, n_kv_heads, k_bias.shape[0], 1).squeeze(-1)

    print("\nK weight roundtrip test:")
    print(f"Max difference in weights: {(k_weight - k_restored).abs().max().item():.2e}")
    print(f"Max difference in biases: {(k_bias - k_bias_restored).abs().max().item():.2e}")


if __name__ == "__main__":
    test_permute_roundtrip()
