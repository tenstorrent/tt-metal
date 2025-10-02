AlexP Examples
==============

- alexp_nlp_concat_heads_boltz_cpp: C++ example invoking ttnn::experimental::nlp_concat_heads_boltz.
- python_nlp_concat_heads_boltz.py: Python example invoking the same op.

Build:
```bash
cmake -S . -B build-cmake -DTT_METAL_BUILD_PROGRAMMING_EXAMPLES=ON
cmake --build build-cmake --target programming_examples -j
```

Run C++ example:
```bash
./build-cmake/programming_examples/alexp_examples/alexp_nlp_concat_heads_boltz_cpp
```

Run Python example:
```bash
python3 programming_examples/alexp_examples/python_nlp_concat_heads_boltz.py
```
