# Best Practices for C++20 Repository

## 1. Pass Complex Types by Const References

### Practice
Always pass all complex types (e.g., vectors, tensors) by const references.

### Explanation
Passing complex types such as vectors and tensors by const reference avoids unnecessary copying of data. This not only enhances performance by reducing overhead but also ensures that the function does not modify the original data.

### Motivation
- **Performance**: Avoids the cost of copying large data structures.
- **Safety**: Prevents unintended modifications to the data.

### Example
```
void write_buffer(queue_id cq_id, Tensor& dst, std::vector<std::shared_ptr<void>> src, const std::optional<std::size_t> transfer_size = std::nullopt); // Wrong!
void write_buffer(queue_id cq_id, Tensor& dst, const std::vector<std::shared_ptr<void>>& src, const std::optional<std::size_t>& transfer_size = std::nullopt); // Right!
```

## 2. Use `std::span` for Input Parameters

### Practice
Consider using `std::span` as input instead of `std::vector`. This allows `std::array` to be used as an argument as well.

### Explanation
`std::span` is a lightweight view over a contiguous sequence of objects, such as arrays and vectors. It provides a safe and flexible way to handle array-like data structures without copying them.

### Motivation
- **Flexibility**: Enables functions to accept both `std::vector` and `std::array`.
- **Efficiency**: Eliminates the need for copying data, enhancing performance.
- **Safety**: Provides bounds-checked access to the underlying data.
### Example
```
template <typename T>
void print_elements(std::span<T> data) {
    for (const auto& element : data) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    std::vector<int> vec = {6, 7, 8, 9, 10};

    // Call print_elements with std::array
    print_elements(arr);

    // Call print_elements with std::vector
    print_elements(vec);

    return 0;
}
```

## 3. Use `std::string_view` Instead of `std::string` or `const char*`

### Practice
Consider using `std::string_view` instead of `std::string` or `const char*`.

### Explanation
`std::string_view` is a lightweight, non-owning view of a string. It allows functions to accept strings without copying them, leading to more efficient code. It is especially useful when you only need to read from the string and do not need to modify it.

### Motivation
- **Performance**: Avoids unnecessary copying of strings, improving performance.
- **Flexibility**: Can be used with different types of string-like data, enhancing versatility.
- **Efficiency**: Reduces memory usage and increases efficiency by providing a non-owning view.

### Example
```
#include <iostream>
#include <string>
#include <string_view>

// Function to count the number of vowels in a string using string_view
size_t count_vowels(std::string_view str) {
    size_t count = 0;
    for (char c : str) {
        switch (c) {
            case 'a': case 'e': case 'i': case 'o': case 'u':
            case 'A': case 'E': case 'I': case 'O': case 'U':
                ++count;
                break;
            default:
                // do nothing
                break;
        }
    }
    return count;
}

int main() {
    std::string myString = "Hello, world!";
    const char* cString = "Example string";  // C-style string

    // Using string_view on a std::string
    std::cout << "Vowels in '" << myString << "': " << count_vowels(myString) << std::endl;
    // Using string_view on a C-style string
    std::cout << "Vowels in '" << cString << "': " << count_vowels(cString) << std::endl;

    return 0;
}
```


## 4. Avoid Dynamic Allocations in Frequently Called Functions

### Practice
Try to avoid dynamic allocations in functions that are called every time a user runs an operation. If the vector size is known, use `std::array` or `std::tuple`. `ttnn::small_vector`(not implemented yet) is a good replacement for `std::vector` in 99% of use cases.

### Explanation
Dynamic allocations can be costly in terms of performance, especially in functions that are called frequently. Using fixed-size containers like `std::array` or `std::tuple` can significantly reduce overhead. `ttnn::small_vector` provides a performance-optimized alternative to `std::vector` for most scenarios.

### Motivation
- **Performance**: Reduces the overhead associated with dynamic memory allocation.
- **Predictability**: Improves predictability of memory usage and performance.
- **Efficiency**: Enhances overall efficiency by leveraging fixed-size containers or optimized small vectors.

### Example
```
// Allocating on heap
std::vector<int> dynamicVector = {1, 2, 3, 4, 5};

// No heap allocation
std::array<int, 5> fixedArray = {1, 2, 3, 4, 5};
```

## 5. Avoid Using `.at()` for Vector Access

### Practice
Avoid calling the `.at()` function on vectors and other stl containers. Perform all necessary checks before using the vector and use `TT_ASSERT` or `TT_FATAL`. The `.at()` function throws exceptions without context, which can be frustrating for users. Overall, avoid STL calls that throw exceptions. All inputs should be validated on our side.

### Explanation
The `.at()` function performs bounds checking and throws an exception if the index is out of range. However, the exceptions lack detailed context, making them hard to debug. By using `TT_ASSERT`, `TT_FATAL` or `TT_THROW`, you can provide clearer error messages and avoid exceptions.

### Motivation
- **User Experience**: Provides more informative error messages, improving user experience.
- **Debugging**: Easier to debug issues with clear assertions and fatal errors.
- **Performance**: Avoids the overhead associated with exception handling.

### Example
```
for (int i = 0; i < fp32_vec.size(); i++) {
  fp32_vec.at(i) = rand_float() + offset; // Very bad, breaks a lot of optimiations. Additional check.
}
for (int i = 0; i < fp32_vec.size(); i++) {
  fp32_vec[i] = rand_float() + offset; // Good!
```


## 6. Avoid `std::move` When Returning from Functions

### Practice
Don't use `std::move` when returning from a function, as it breaks Return Value Optimization (RVO).

### Explanation
RVO is an optimization technique that eliminates unnecessary copying or moving of objects when they are returned from a function. Using `std::move` can prevent the compiler from applying RVO, resulting in less efficient code.

### Motivation
- **Performance**: Ensures that RVO can be applied, minimizing unnecessary copying or moving of objects.
- **Efficiency**: Maintains optimal performance by allowing the compiler to perform return value optimizations.

### Example
https://godbolt.org/z/enTqadjMz

## 7. Avoid `const T&&` or `const auto&&` and Returning `const` Values

### Practice
Never use `const T&&` or `const auto&&`, and never return `const` values from functions.

### Explanation
Using `const` rvalue references (`const T&&` or `const auto&&`) and returning `const` values from functions can prevent the compiler from optimizing the code. These practices block the use of move constructors, which are crucial for efficient resource management and performance optimization.

### Motivation
- **Performance**: Enables the use of move constructors, enhancing performance.
- **Optimization**: Allows the compiler to optimize code more effectively.
- **Efficiency**: Facilitates efficient resource management by leveraging move semantics.

### Example
https://godbolt.org/z/PsK4vahMr

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

### Example
https://stackoverflow.com/questions/28627348/why-does-a-throwing-move-constructor-result-in-copying-instead-of-moving-where-a

## 10. Use the Copy-and-Swap Idiom

### Practice
Use the Copy-and-Swap idiom to avoid duplicating code between different constructors and assignment operators.

### Explanation
The Copy-and-Swap idiom is a robust and elegant method to implement copy assignment operators. It leverages the copy constructor and the swap method to provide strong exception safety and reduce code duplication.

### Example 
https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom


## 11. Avoid Global Classes, Especially with Mutexes or Locks

### Practice
Avoid creating global classes, especially those with mutexes or other locks, except for specific debug purposes.

### Explanation
Global classes can lead to issues with concurrency and resource management, especially when they involve mutexes or other synchronization mechanisms. Such global resources can create hidden dependencies and make the code harder to reason about and maintain.

### Motivation
- **Concurrency Safety**: Reduces the risk of deadlocks and race conditions.
- **Maintainability**: Avoids hidden dependencies and makes the code easier to understand and maintain.
- **Predictability**: Improves the predictability of resource management and execution flow.

## 12. Move Function  Implementations to `.cpp` Files

### Practice
Move function and method implementations from header files to `.cpp` files. Template heavy functions/methods implementations could be moved to the '_inl.hpp' file which should be included in the end of the header.

### Explanation
Implementing functions in `.cpp` files rather than in headers reduces compilation dependencies and improves encapsulation. It also leads to faster compile times and reduces the chances of multiple definition errors.

### Motivation
- **Compilation Time**: Reduces compilation times by minimizing dependencies.
- **Encapsulation**: Improves encapsulation by hiding implementation details.
- **Maintainability**: Simplifies code maintenance by separating interface from implementation.

## 13. Avoid `using namespace` in Headers

### Practice
Never write `using namespace` in header files.
The worst example which should be avoided in 100% cases is `using namespace std;`.

### Explanation
Including `using namespace` in header files can lead to namespace pollution and conflicts, making the codebase harder to manage and increasing the risk of name collisions.

### Motivation
- **Namespace Pollution**: Prevents namespace pollution and reduces the risk of name collisions.
- **Code Clarity**: Improves code clarity by explicitly specifying the namespace.
- **Maintainability**: Enhances maintainability by avoiding unintended interactions between different parts of the code.

## 14. Avoid Bool Arguments in APIs

### Practice
Avoid using plain bool arguments in APIs, especially when the meaning isn't clear at the call site. Instead, use enum class to provide context.

### Explanation
Using a bool argument in a function call can make the code less readable and harder to maintain, as the purpose of the true or false value is often unclear. By using an enum class, you make the intention explicit, improving code clarity.

### Example
Avoid:
```cpp
tensor = tt::tt_metal::tilize_with_val_padding(tensor, output_shape, 0, output_memory_config, dtype, true);
```
Prefer:
```cpp
enum class ThreadingOption { SingleCore, MultiCore };
tensor = tt::tt_metal::tilize_with_val_padding(tensor, output_shape, 0, output_memory_config, dtype, ThreadingOption::MultiCore);
```
Also consider giving enums power-of-2 values to pass them all as a single argument, e.g. 
```cpp
Options::FOO | Options::BAR
```

### Motivation
- **Readability:** Enhances readability by making the purpose of the argument explicit.
- **Maintainability:** Reduces the likelihood of misusing the API, making it easier to maintain and extend.
- **User Experience:** Helps in winning over users by providing a clearer, more intuitive API.

## 15. Initialize Primitive Types on Declaration
### Practice
Always initialize primitive types (e.g., size_t, int, float, bool, pointers) at the point of declaration.

### Explanation
Forgetting to initialize primitive types can lead to unpredictable behavior, as uninitialized variables may contain garbage values. This can cause hard-to-debug issues, especially in large codebases.

### Example
Avoid:
```cpp
struct PadDimension {
    std::size_t front;
    std::size_t back;

    static constexpr auto attribute_names = std::make_tuple("front", "back");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->front), std::cref(this->back)); }
};
```
Prefer:
```cpp
struct PadDimension {
    std::size_t front = 0;
    std::size_t back = 0;

    static constexpr auto attribute_names = std::make_tuple("front", "back");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->front), std::cref(this->back)); }
};
```
Motivation
- **Bug Prevention:** Reduces the risk of bugs due to uninitialized variables.
- **Code Safety:** Ensures that all variables have a known value, leading to safer and more predictable code.
- **Ease of Review:** Simplifies code reviews by making initialization explicit.

## 16. Use Early Exit for Contract Checks
### Practice
Use early exit strategies when performing contract checks or validations at the start of a function.

### Explanation
Placing contract checks at the start of a function and returning early if they fail simplifies the function logic, reducing nesting and improving readability.
Keep in mind, compiler knows that early checks is the slow path and branch prediction works better this way.

### Example
Avoid:
```cpp
void doSomething(...) {
    if (contractCheck) {
        // Do a lot of things
        // More complex logic
    }
}
```
Prefer:
```cpp
void doSomething(...) {
    if (!contractCheck) 
        return;

    // Do a lot of things
    // More complex logic
}
```
### Motivation
- **Code Clarity:** Improves code clarity by reducing unnecessary nesting.
- **Maintainability:** Makes the code easier to maintain by focusing on the main logic once preconditions are validated.
- **Efficiency:** Potentially improves performance by avoiding unnecessary processing when contract conditions aren't met.
