Here's an introductory section for a matrix multiplication lab in .rst format:

Lab 1: Single Core Matrix Multiplication
========================================

Introduction
------------

Matrix multiplication is a fundamental operation in linear algebra with widespread applications in scientific computing, machine learning, and computational graphics.
The standard matrix multiplication algorithm transforms two input matrices into a third matrix using a simple computational procedure.

Basic Algorithm
---------------

Given two matrices A of shape (M, K) and B of shape (K, N), the resulting matrix C will have shape (M, N).
The classical matrix multiplication algorithm can be expressed using a triple-nested loop structure,
where each element of the resulting matrix is computed as the dot product of the corresponding row of A and the corresponding column of B:

.. code-block:: cpp

   for (int i = 0; i < M; i++) {
       for (int j = 0; j < N; j++) {
           C[i][j] = 0.0;
           for (int k = 0; k < K; k++) {
               C[i][j] += A[i][k] * B[k][j];
           }
       }
   }

Computational Complexity
------------------------

The naive implementation has a time complexity of O(M*N*K)


Linear Transformation and Computational Flexibility
===================================================

Matrix Multiplication as a Linear Function
------------------------------------------

Matrix multiplication represents a linear transformation where the computation can be viewed as a sequence of dot products.
This linearity allows significant computational flexibility, such as:

1. Loop Reordering: Changing the order of loops does not impact the result, but can impact performance on architectures with cache hierarchies.
2. Loop Tiling: A subset of the resulting matrix can be calculated at a time, improving cache locality and performance.
3. Parallelization: Different matrix regions can be computed concurrently

Note that operation ordering can impact result when floating point operations are not associative, which is the case for most modern architectures.
In this lab, we will ignore this issue, but you should be aware of it, especially in cases when the values in matrices are of significantly different orders of magnitude.


Loop Tiling
-----------

Loop tiling (a.k.a. loop blocking) is a loop transformation technique that divides the loop into smaller chunks.
On architectures with cache hierarchies, this can significantly improve performance.
On the other hand, some architectures offer hardware support for vector or matrix operations of limited vector/matrix sizes.
When targeting such architectures, loop tiling can be used to divide work into smaller chunks that map to the underlying hardware.

Simple Tiling Example
^^^^^^^^^^^^^^^^^^^^^

Original Doubly Nested Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Original loop without tiling
   for (int i = 0; i < M; i++)
       for (int j = 0; j < N; j++)
           some_computation(i, j);

Tiled Version
~~~~~~~~~~~~~

.. code-block:: cpp

   // Tiled version with M-N blocking
   int M_TILE_SIZE = 32;
   int N_TILE_SIZE = 32;

   // Assumes M and N are divisible by tile sizes
   int num_row_tiles = M / M_TILE_SIZE;
   int num_col_tiles = N / N_TILE_SIZE;

   for (int row_tile = 0; row_tile < num_row_tiles; row_tile++)
       for (int col_tile = 0; col_tile < num_col_tiles; col_tile++)
           // Inner tile computation
           for (int row = row_tile * M_TILE_SIZE; row < (row_tile + 1) * M_TILE_SIZE; row++)
               for (int col = col_tile * N_TILE_SIZE; col < (col_tile + 1) * N_TILE_SIZE; col++)
                   some_computation(row, col);


Tiled Matrix Multiplication
===========================

In this part of the lab, you will implement two versions of matrix multiplication: a straightforward triply-nested loop, and a tiled version.
The triply-nested loop version is simply a reference implementation provided at the beginning of the lab.
The tiled version should be implemented as follows:

1. Input to the matrix multiplication should be a vector to ensure data is contiguous in memory.
1. Create a function ``tile_matmul`` that multiplies a single tile
2. Implement a main matrix multiplication function using tiling and then calling ``tile_matmul`` for each tile.
3. Allow parameterization of tile size
4. Measure performance and compare it to the triply-nested loop version


.. code-block:: cpp

    // Single tile matrix multiplication
    void tile_matmul(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::vector<float>& C,
        int K,
        int N,
        int row_offset, // Tile starting indices
        int col_offset,
        int k_offset,
        int TH, // Tile height
        int TW // Tile width
    ) {
        // Implement multiplication of TH x TW tile by TW x TH matrix to produce TH x TH result.
        // Use the starting indices to index into matrices A, B and C.
        // Tile in A begins at (row_offset, k_offset) and has size TH x TW.
        // Tile in B begins at (k_offset, col_offset) and has size TW x TH.
        // Resulting tile in C begins at (row_offset, col_offset) and has size TH x TH.
        // Accumulate the result into C (assume it is initialized to 0).
        // Hint: your code will be a lot simpler if you create a helper function get_idx to index
        // into a matrix using row and column coordinates and number of columns.
    }

    std::vector<float> tiled_matrix_multiply(
        const std::vector<float>& A,
        const std::vector<float>& B,
        int M, // Full matrix dimensions
        int K,
        int N,
        int TH, // Tile height
        int TW // Tile width
    ) {
        // Implement full matrix multiplication using tiling
    }
