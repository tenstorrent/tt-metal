# Guidelines for Writing Effective Error Messages ✍️
It's important to remember the impact that clear and informative error messages can have on debugging and maintenance. A well-crafted error message can save hours of troubleshooting and make our codebase more user-friendly, especially for those who may be less familiar with the inner workings of the system.

## Common Mistakes
1. **Being Too Vague:** Avoid generic messages like "An error occurred" or "Invalid input." These do not help in identifying the root cause.
2. **Not Including Variable Values:** Forgetting to include the actual values that caused the error can make debugging significantly harder.
3. **Overloading Messages:** While it’s important to be informative, avoid making the message too long or cluttered. Keep it concise and to the point.
4. **Assuming Prior Knowledge:** Don’t assume that the person encountering the error knows as much about the code as you do. Provide enough context to make the error understandable on its own.
5. **Not Making the Message Actionable:** Ensure that the error message provides clear guidance on what needs to be done to resolve the issue. Simply stating what went wrong without suggesting how to fix it can leave the user frustrated and confused.

## Be Specific
Always include the actual values and conditions that caused the error. This helps immediately identify the issue without needing to dig into the code.
Instead of writing:
```cpp
TT_FATAL(input_shape.rank() == 3, "Invalid input tensor dimensions.");
```
Write:
```cpp
TT_FATAL(input_shape.rank() == 3, fmt::format("Invalid input tensor: expected 3 dimensions, but found {}.", input_shape.rank()));
```

## Explain the Issue
Provide a brief explanation of why the error occurred or why the condition is important. This helps users understand the context of the error.

Instead of:
```cpp
TT_FATAL(!input_tensor_kv.has_value(), "KV tensor cannot be passed in when sharded.");
```
Write:
```cpp
TT_FATAL(!input_tensor_kv.has_value(), "Invalid operation: KV tensor should not be provided when the input tensor is sharded. Please ensure that the KV tensor is only used in non-sharded configurations.");
```

## Avoid Ambiguity
Make sure the error message clearly states what is wrong. Avoid vague language that leaves room for interpretation.
Instead of:
```cpp
TT_FATAL(error_code == SUCCESS, "Operation failed.");
```
Write:
```cpp
TT_FATAL(error_code == SUCCESS, fmt::format("Operation failed with error code {}. Ensure that the previous steps were successful.", error_code));
```

## Avoid Redundancy
Don’t repeat information that’s already obvious or present in the code. Focus on what adds value to the message.
Avoid:
```cpp
TT_FATAL(ptr != nullptr, "Pointer must not be null.");
```
Instead, enhance the message by providing context:
```cpp
TT_FATAL(ptr != nullptr, "Failed to allocate memory: pointer is null.");
```

## Make the Message Actionable
Ensure that the error message provides clear guidance on what needs to be done to resolve the issue. Simply stating what went wrong without suggesting how to fix it can leave the user frustrated and confused.
Instead of:
```cpp
TT_FATAL(head_size % TILE_WIDTH != 0, "Head size is invalid.");
```
Write:
```cpp
TT_FATAL(head_size % TILE_WIDTH != 0, fmt::format("Invalid head size: {}. The head size must be a multiple of tile width ({}). Please adjust the dimensions accordingly.", head_size, TILE_WIDTH));
```

## Example of a well-constructed error message:
```cpp
TT_FATAL(head_size % TILE_WIDTH == 0,
         fmt::format("Invalid head size: {}. The head size must be a multiple of the tile width ({}). Please adjust the dimensions accordingly.", 
                     head_size, TILE_WIDTH));
```
This message clearly states the problem (head size not being a multiple of the tile width), includes the actual value of head_size, and offers guidance on what needs to be done to fix it.
