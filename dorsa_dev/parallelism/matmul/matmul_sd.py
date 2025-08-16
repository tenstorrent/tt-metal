# Matmul, single-device

import torch
import ttnn

# shape of tensors
A_rows, A_cols, B_rows, B_cols = 1, 2, 2, 1

assert A_cols == B_rows, "A's columns must equal B's rows"

# seed
torch.manual_seed(0)

A = torch.randn(A_rows, A_cols)
B = torch.randn(B_rows, B_cols)

C_torch = torch.matmul(A, B)

# print(C_torch)


# === Matmul, single-device from scratch ===

C = torch.zeros(A_rows, B_cols)

# move across rows of A
for i in range(len(A)):
    # move across columns of B
    for j in range(len(B[0])):
        # move across elements in A's row and B's column
        for k in range(len(A[0])):
            C[i][j] += A[i][k] * B[k][j]

print("C torch:", C_torch)
print("C from scratch:", C)

assert C_torch.shape == C.shape, "C_torch and C should have the same shape"
assert torch.allclose(C_torch, C, atol=1e-4), "C_torch and C should be close"

print("Correct: C_torch and C are the same!")
