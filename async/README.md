# New Async

## Pitfalls of current async

- Not clear when async is used by looking at the code
- Very coupled implementation. Tensor is unrecognizable from a simple class it used to be before.
- Op writers need to explicily call launch_op. And they also are responsible for providing the right number of output tensors before even running the op
- Model writers don't know if they are running async or not just based on the data structures they are using

## Proposal

The proposed solution is to make `async` to be written using C++ way of doing things and to leverage `std::promise` and `std::shared_future`.

Async mode would be entered by simply making an input to an op `async`:

#### Not async
```cpp
Tensor activations{activations_shape, dtype, device};
Tensor weights{weights_shape, dtype, device};
Tensor output = ttnn::matmul(activations, weights);
std::tuple<Tensor, Tensor> slices = ttnn::split(output, dim, num_slices);
```

#### Async
```cpp
Tensor activations{activations_shape, dtype, device};
Tensor weights{weights_shape, dtype, device};
std::shared_future<Tensor> async_activations = ttnn::to_async(activations);
std::shared_future<Tensor> async_output = ttnn::matmul(activations, weights);
std::shared_future<std::tuple<Tensor, Tensor>> async_slices = ttnn::split(async_output, dim, num_slices);


// do something else

// Wait for slices to be created
std::tuple<Tensor, Tensor> slices = async_slices.get();
ttnn::synchronize_device(device);

```

Once async mode is entered, main thread will push all operations to the dispatcher thread work queue, which will free up the main thread to do something else.

In addition to dispatcher thread, there are device worker threads (just like now).
And each operation that runs will dispatch the creation of tensors and programs onto device worker threads.

Here is the diagram of all threads and their relations:

![alt text](diagram.png)

The threads are controlled by the infra and if something needs to be put into different boxes, it can be done by the infra without model and op writers having to worry about anything (in most cases)


> **_NOTE:_**  operations.hpp is written with a new concept of primitive vs device operations. It's not necessary to do that in actual `ttnn` code to make this version of async work exactly same way as shown.

## Further Work

This might not be the most optimal way to split work on different threads and could be improved. For example, we could add more threads to each device to remove some of the current dependencies.

# Running the examples

```bash
clear && clang++-17 main.cpp -std=c++20 -stdlib=libc++ -pthread -O3 -o main && ./main
```
