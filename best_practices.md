# Best Practices for C++20 Repository

## 1. Pass Complex Types by Const References

### Practice
Always pass all complex types (e.g., vectors, tensors) by const references.

### Explanation
Passing complex types such as vectors and tensors by const reference avoids unnecessary copying of data. This not only enhances performance by reducing overhead but also ensures that the function does not modify the original data.

### Motivation
- **Performance**: Avoids the cost of copying large data structures.
- **Safety**: Prevents unintended modifications to the data.

## 2. Use `std::span` for Input Parameters

### Practice
Consider using `std::span` as input instead of `std::vector`. This allows `std::array` to be used as an argument as well.

### Explanation
`std::span` is a lightweight view over a contiguous sequence of objects, such as arrays and vectors. It provides a safe and flexible way to handle array-like data structures without copying them.

### Motivation
- **Flexibility**: Enables functions to accept both `std::vector` and `std::array`.
- **Efficiency**: Eliminates the need for copying data, enhancing performance.
- **Safety**: Provides bounds-checked access to the underlying data.

## 3. Use `std::string_view` Instead of `std::string` or `const char*`

### Practice
Consider using `std::string_view` instead of `std::string` or `const char*`.

### Explanation
`std::string_view` is a lightweight, non-owning view of a string. It allows functions to accept strings without copying them, leading to more efficient code. It is especially useful when you only need to read from the string and do not need to modify it.

### Motivation
- **Performance**: Avoids unnecessary copying of strings, improving performance.
- **Flexibility**: Can be used with different types of string-like data, enhancing versatility.
- **Efficiency**: Reduces memory usage and increases efficiency by providing a non-owning view.


## 4. Avoid Dynamic Allocations in Frequently Called Functions

### Practice
Try to avoid dynamic allocations in functions that are called every time a user runs an operation. If the vector size is known, use `std::array` or `std::tuple`. `ttnn::small_vector` is a good replacement for `std::vector` in 99% of use cases.

### Explanation
Dynamic allocations can be costly in terms of performance, especially in functions that are called frequently. Using fixed-size containers like `std::array` or `std::tuple` can significantly reduce overhead. `ttnn::small_vector` provides a performance-optimized alternative to `std::vector` for most scenarios.

### Motivation
- **Performance**: Reduces the overhead associated with dynamic memory allocation.
- **Predictability**: Improves predictability of memory usage and performance.
- **Efficiency**: Enhances overall efficiency by leveraging fixed-size containers or optimized small vectors.

## 5. Avoid Using `.at()` for Vector Access

### Practice
Avoid calling the `.at()` function on vectors and other stl containers. Perform all necessary checks before using the vector and use `TT_ASSERT` or `TT_FATAL`. The `.at()` function throws exceptions without context, which can be frustrating for users. Overall, avoid STL calls that throw exceptions. All inputs should be validated on our side.

### Explanation
The `.at()` function performs bounds checking and throws an exception if the index is out of range. However, the exceptions lack detailed context, making them hard to debug. By using `TT_ASSERT`, `TT_FATAL` or `TT_THROW`, you can provide clearer error messages and avoid exceptions.

### Motivation
- **User Experience**: Provides more informative error messages, improving user experience.
- **Debugging**: Easier to debug issues with clear assertions and fatal errors.
- **Performance**: Avoids the overhead associated with exception handling.

## 6. Avoid `std::move` When Returning from Functions

### Practice
Don't use `std::move` when returning from a function, as it breaks Return Value Optimization (RVO).

### Explanation
RVO is an optimization technique that eliminates unnecessary copying or moving of objects when they are returned from a function. Using `std::move` can prevent the compiler from applying RVO, resulting in less efficient code.

### Motivation
- **Performance**: Ensures that RVO can be applied, minimizing unnecessary copying or moving of objects.
- **Efficiency**: Maintains optimal performance by allowing the compiler to perform return value optimizations.
