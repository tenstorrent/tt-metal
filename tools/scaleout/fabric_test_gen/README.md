

## Schema Setup
- Compile the `.proto` schema into Python bindings using `protoc`, for example:

  ```bash
  protoc --python_out=. mesh_graph_descriptor.proto
## Use
- Pass `.textproto` file as the MGD to be parsed
- Other optional arguments to configure worker cores, noc_type etc.
- Output `.yaml` can be used as the `--test_config` for fabric tests

`python3 main.py -f path/to/.textproto [options]`
