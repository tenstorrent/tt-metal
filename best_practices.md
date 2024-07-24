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

## 7. Avoid `const T&&` or `const auto&&` and Returning `const` Values

### Practice
Never use `const T&&` or `const auto&&`, and never return `const` values from functions.

### Explanation
Using `const` rvalue references (`const T&&` or `const auto&&`) and returning `const` values from functions can prevent the compiler from optimizing the code. These practices block the use of move constructors, which are crucial for efficient resource management and performance optimization.

### Motivation
- **Performance**: Enables the use of move constructors, enhancing performance.
- **Optimization**: Allows the compiler to optimize code more effectively.
- **Efficiency**: Facilitates efficient resource management by leveraging move semantics.

## 8. Use `std::move` with `emplace_back` and Prefer `push_back` When Appropriate

### Practice
The `emplace_back` call takes the same parameters as the constructor. Passing a type by value calls the copy constructor, negating the benefits of `emplace_back`. Use `std::move` when necessary, and consider using `push_back` with the same efficiency.

### Explanation
`emplace_back` constructs an element in place, avoiding unnecessary copies or moves when the arguments match the constructor parameters. However, if parameters are passed by value, a copy constructor is invoked. Using `std::move` ensures that resources are moved rather than copied. In some cases, `push_back` can be just as efficient and more straightforward.

### Motivation
- **Efficiency**: Ensures resources are moved, not copied, maintaining performance.
- **Clarity**: Using `push_back` can be simpler and equally efficient in certain scenarios.
- **Optimization**: Maximizes the benefits of in-place construction and move semantics.

## 9. Move constructor should not throw. Mark Move Constructors as `noexcept`

### Practice
Move constructors should be marked as `noexcept`, otherwise many STL optimizations will not be applied.

### Explanation
The `noexcept` specifier indicates that a function does not throw exceptions. Marking move constructors as `noexcept` allows the Standard Library to perform various optimizations, such as using move operations in containers more effectively.

### Motivation
- **Performance**: Enables STL optimizations that rely on `noexcept` guarantees.
- **Safety**: Clearly indicates that move operations are safe and will not throw exceptions.
- **Efficiency**: Enhances the performance of STL containers by allowing move operations.

## 10. Use the Copy-and-Swap Idiom

### Practice
Use the Copy-and-Swap idiom to avoid duplicating code between different constructors and assignment operators.

### Explanation
The Copy-and-Swap idiom is a robust and elegant method to implement copy assignment operators. It leverages the copy constructor and the swap method to provide strong exception safety and reduce code duplication.

### Motivation

## 11. Avoid Global Classes, Especially with Mutexes or Locks

### Practice
Avoid creating global classes, especially those with mutexes or other locks, except for specific debug purposes.

### Explanation
Global classes can lead to issues with concurrency and resource management, especially when they involve mutexes or other synchronization mechanisms. Such global resources can create hidden dependencies and make the code harder to reason about and maintain.

### Motivation
- **Concurrency Safety**: Reduces the risk of deadlocks and race conditions.
- **Maintainability**: Avoids hidden dependencies and makes the code easier to understand and maintain.
- **Predictability**: Improves the predictability of resource management and execution flow.

## 12. Move Function Implementations to `.cpp` Files

### Practice
Move function implementations from header files to `.cpp` files.

### Explanation
Implementing functions in `.cpp` files rather than in headers reduces compilation dependencies and improves encapsulation. It also leads to faster compile times and reduces the chances of multiple definition errors.

### Motivation
- **Compilation Time**: Reduces compilation times by minimizing dependencies.
- **Encapsulation**: Improves encapsulation by hiding implementation details.
- **Maintainability**: Simplifies code maintenance by separating interface from implementation.

## 13. Avoid `using namespace std;` in Headers

### Practice
Never write `using namespace std;` in header files.

### Explanation
Including `using namespace std;` in header files can lead to namespace pollution and conflicts, making the codebase harder to manage and increasing the risk of name collisions.

### Motivation
- **Namespace Pollution**: Prevents namespace pollution and reduces the risk of name collisions.
- **Code Clarity**: Improves code clarity by explicitly specifying the namespace.
- **Maintainability**: Enhances maintainability by avoiding unintended interactions between different parts of the code.

