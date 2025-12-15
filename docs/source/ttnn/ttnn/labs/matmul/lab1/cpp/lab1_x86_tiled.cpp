#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

// Define tile parameters (Can be changed or passed as template arguments)
// M and N must be divisible by TILE_HEIGHT
// K must be divisible by TILE_WIDTH
const int TILE_HEIGHT = 16;
const int TILE_WIDTH = 16;

// Simple triple-loop matrix multiplication
std::vector<double> simple_matrix_multiply(
    const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) {
    std::vector<double> C(M * N, 0.0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }

    return C;
}

// Helper to calculate 1D index from 2D coordinates
// row_stride corresponds to the number of columns in the full matrix
inline int get_idx(int row, int col, int row_stride) { return row * row_stride + col; }

/**
 * 2. Create a function tile_matmul that multiplies a single tile
 * * This function computes a partial matrix multiplication for a specific tile.
 * It accumulates results into the C matrix.
 * * @param A        Input vector for Matrix A
 * @param B        Input vector for Matrix B
 * @param C        Output vector for Matrix C (result accumulated here)
 * @param M, K, N  Global dimensions of the matrices
 * @param row_offset A/C  The global row index where this tile starts
 * @param col_offset B/C  The global col index where this tile starts
 * @param k_offset        The global K index where the calculation strip starts
 */
void tile_matmul(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    int K,
    int N,  // We need K for A's stride, N for B and C's stride
    int row_offset,
    int col_offset,
    int k_offset) {
    // Iterate over the TILE_HEIGHT (Rows of A, Rows of C)
    for (int i = 0; i < TILE_HEIGHT; ++i) {
        // Iterate over the TILE_HEIGHT (Cols of B, Cols of C)
        // Note: User specified M and N divisible by TILE_HEIGHT, implying square output tiles
        for (int j = 0; j < TILE_HEIGHT; ++j) {
            // We accumulate the dot product for the specific tile elements
            double sum = 0.0;

            // Iterate over the TILE_WIDTH (Cols of A, Rows of B)
            for (int k = 0; k < TILE_WIDTH; ++k) {
                // Global coordinates
                int global_row_A = row_offset + i;
                int global_col_A = k_offset + k;

                int global_row_B = k_offset + k;
                int global_col_B = col_offset + j;

                // Access data using 1D indices
                double val_A = A[get_idx(global_row_A, global_col_A, K)];
                double val_B = B[get_idx(global_row_B, global_col_B, N)];

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
 * 3. Implement a main matrix multiplication function using tiling
 */
std::vector<double> tiled_matrix_multiply(
    const std::vector<double>& A, const std::vector<double>& B, int M, int K, int N) {
    // 1. Input ensures data is contiguous (std::vector)
    // Validate assumptions
    assert(M % TILE_HEIGHT == 0 && "M must be divisible by TILE_HEIGHT");
    assert(N % TILE_HEIGHT == 0 && "N must be divisible by TILE_HEIGHT");
    assert(K % TILE_WIDTH == 0 && "K must be divisible by TILE_WIDTH");

    // Initialize Result Matrix C with zeros
    std::vector<double> C(M * N, 0.0);

    // 4 & 5. Loop over the matrix in steps of tile sizes

    // Iterate over rows of C (M dimension)
    for (int i = 0; i < M; i += TILE_HEIGHT) {
        // Iterate over columns of C (N dimension)
        for (int j = 0; j < N; j += TILE_HEIGHT) {
            // Iterate over the shared dimension (K dimension)
            // We move across A and down B
            for (int k = 0; k < K; k += TILE_WIDTH) {
                // Process the specific tile
                tile_matmul(A, B, C, K, N, i, j, k);
            }
        }
    }

    return C;
}

// Helper to print matrices
void print_matrix(const std::vector<double>& mat, int rows, int cols, const std::string& name) {
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
    const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C, int M, int K, int N) {
    // Create a reference implementation to compare against
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> ref_C = simple_matrix_multiply(A, B, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto tiled_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Reference implementation took " << tiled_duration.count() << " microseconds" << std::endl;

    // Compare results with some tolerance
    const double EPSILON = 1e-6;
    for (size_t i = 0; i < C.size(); i++) {
        if (std::abs(C[i] - ref_C[i]) > EPSILON) {
            std::cerr << "Mismatch at index " << i << ": " << C[i] << " vs " << ref_C[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // 4. Setup Dimensions
    // Ensure these fit the divisibility rules with TILE_HEIGHT=2, TILE_WIDTH=2
    int M = 1600;
    int K = 1600;
    int N = 1600;

    // Create Dummy Data (Linear sequence for easy verification)
    std::vector<double> A(M * K);
    std::vector<double> B(K * N);

    for (int i = 0; i < M * K; ++i) {
        A[i] = i + 1;  // 1, 2, 3...
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (i + 1) * 2;  // 2, 4, 6...
    }

    // Perform Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> tiled_C = tiled_matrix_multiply(A, B, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto tiled_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Tiled implementation took " << tiled_duration.count() << " microseconds" << std::endl;

    // Output Results
    std::cout << "Tiling Params: Height=" << TILE_HEIGHT << ", Width=" << TILE_WIDTH << "\n" << std::endl;
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
