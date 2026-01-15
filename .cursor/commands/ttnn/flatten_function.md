# Analyze the call tree of a given function and produce a flattened version

Flattened code does not have calls to other functions except lowest level compiler intrinsics.

## How to do that
Fetch the implementation code of a given function.
For each function it calls, find it and fetch the code of that function.
Attempt to put it all into a single function.

It is good to keep in mind constrains when dealing with templated code.
Try to flatten given specific constrains as this helps to elimiante whole branches of irrelevant code.
