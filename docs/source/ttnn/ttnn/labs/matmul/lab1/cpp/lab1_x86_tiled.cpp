#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

// Define tile parameters (Can be changed or passed as template arguments)
// TH x TW are dimensions of the output tile (C)
// A tile dimensions: TH x TK
// B tile dimensions: TK x TW
// M must be divisible by TH
// N must be divisible by TW
// K must be divisible by TK
const int TILE_HEIGHT = 16;  // Output tile and A tile height
const int TILE_WIDTH = 16;   // Output tile and B tile width
const int TILE_K = 16;       // A tile width, B tile height

// Simple triple-loop matrix multiplication
std::vector<float> simple_matrix_multiply(
    const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N) {
    std::vector<float> C(M * N, 0.0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    return C;
}

// Helper to calculate 1D index from 2D coordinates.
// columns corresponds to the number of columns in the full matrix
inline int get_idx(int row, int col, int columns) { return row * columns + col; }

/**
 * Function that multiplies a single tile.
 * It accumulates results into the C matrix.
 * @param A        Input vector for Matrix A
 * @param B        Input vector for Matrix B
 * @param C        Output vector for Matrix C (result accumulated here)
 * @param K        Number of columns in A (and rows in B)
 * @param N        Number of columns in B (and C)
 * @param row_offset A/C  The global row index where this tile starts
 * @param col_offset B/C  The global col index where this tile starts
 * @param k_offset        The global K index where the calculation strip starts
 * @param TH       Output tile and A tile height
 * @param TW       Output tile and B tile width
 * @param TK       A tile width, B tile height
 */
void tile_matmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int K,
    int N,  // We need K for A's stride, N for B and C's stride
    int row_offset,
    int col_offset,
    int k_offset,
    int TH,
    int TW,
    int TK) {
    // Iterate over TH: rows of output tile C, rows of A tile
    for (int i = 0; i < TH; ++i) {
        // Iterate over TW: columns of output tile C, columns of B tile
        for (int j = 0; j < TW; ++j) {
            // We accumulate the dot product for the specific tile elements
            float sum = 0.0;

            // Iterate over TK: columns of A tile, rows of B tile
            for (int k = 0; k < TK; ++k) {
                // Global coordinates
                int global_row_A = row_offset + i;
                int global_col_A = k_offset + k;

                int global_row_B = k_offset + k;
                int global_col_B = col_offset + j;

                // Access data using 1D indices
                float val_A = A[get_idx(global_row_A, global_col_A, K)];
                float val_B = B[get_idx(global_row_B, global_col_B, N)];

                sum += val_A * val_B;
            }

            // Accumulate result into C
            // Note: We use += because we might be processing the K dimension in chunks
            int global_row_C = row_offset + i;
            int global_col_C = col_offset + j;
            C[get_idx(global_row_C, global_col_C, N)] += sum;
        }
    }
}

/**
 * Main matrix multiplication function using tiling
 */
std::vector<float> tiled_matrix_multiply(
    const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N, int TH, int TW, int TK) {
    // 1. Input ensures data is contiguous (std::vector)
    // Validate assumptions
    assert(M % TH == 0 && "M must be divisible by TH");
    assert(N % TW == 0 && "N must be divisible by TW");
    assert(K % TK == 0 && "K must be divisible by TK");

    // Initialize Result Matrix C with zeros
    std::vector<float> C(M * N, 0.0);

    // Loop over the matrix in steps of tile sizes

    // Iterate over rows of C (M dimension) - step by TH
    for (int i = 0; i < M; i += TH) {
        // Iterate over columns of C (N dimension) - step by TW
        for (int j = 0; j < N; j += TW) {
            // Iterate over the shared dimension (K dimension) - step by TK
            // We move across A and down B
            for (int k = 0; k < K; k += TK) {
                // Process the specific tile
                // A tile: TH x TK, B tile: TK x TW, C tile: TH x TW
                tile_matmul(A, B, C, K, N, i, j, k, TH, TW, TK);
            }
        }
    }

    return C;
}

// Helper to print matrices
void print_matrix(const std::vector<float>& mat, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(6) << mat[get_idx(i, j, cols)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to verify matrix multiplication
bool verify_matrix_multiply(
    const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, int M, int K, int N) {
    // Create a reference implementation to compare against
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> ref_C = simple_matrix_multiply(A, B, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto tiled_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Reference implementation took " << tiled_duration.count() << " microseconds" << std::endl;

    // Compare results with some tolerance
    constexpr float RELTOL = 0.00001;
    bool passed = true;
    for (size_t i = 0; i < C.size(); i++) {
        float relative_error = std::abs(C[i] - ref_C[i]) / ref_C[i];
        if (relative_error > RELTOL) {
            std::cerr << "Mismatch at index " << i << ": " << C[i] << " vs " << ref_C[i] << std::endl;
            std::cerr << "Expected relative tolerance: " << RELTOL << " actual relative error: " << relative_error
                      << std::endl;
            passed = false;
        }
    }
    return passed;
}

int main() {
    // 4. Setup Dimensions
    // Ensure these fit the divisibility rules with TH=16, TW=16, TK=16
    int M = 1600;
    int K = 1600;
    int N = 1600;

    // Create Dummy Data (Linear sequence for easy verification)
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);

    for (int i = 0; i < M * K; ++i) {
        A[i] = i + 1;  // 1, 2, 3...
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (i + 1) * 2;  // 2, 4, 6...
    }

    std::cout << "Tiling Params: TH=" << TILE_HEIGHT << ", TW=" << TILE_WIDTH << ", TK=" << TILE_K << "\n" << std::endl;
    // Perform Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> tiled_C = tiled_matrix_multiply(A, B, M, K, N, TILE_HEIGHT, TILE_WIDTH, TILE_K);
    auto end = std::chrono::high_resolution_clock::now();

    auto tiled_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Tiled implementation took     " << tiled_duration.count() << " microseconds" << std::endl;

    // Output Results
    //    print_matrix(A, M, K, "A");
    //    print_matrix(B, K, N, "B");
    //    print_matrix(tiled_C, M, N, "Tiled C");

    // Verify correctness
    if (verify_matrix_multiply(A, B, tiled_C, M, K, N)) {
        std::cout << "Matrix multiplication verified successfully!" << std::endl;
    } else {
        std::cerr << "Verification failed!" << std::endl;
        return 1;
    }
    return 0;
}
