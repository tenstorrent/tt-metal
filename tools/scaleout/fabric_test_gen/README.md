

## Schema Setup
- Compile a `.proto` file using `protoc` for python
- `import` into some `.py` file and use to parse a `.textproto` and enforce the schema

## Use
- Pass `.textproto` file as the MGD to be parsed
- Other optional arguments to configure worker cores, noc_type etc.
- Output `.yaml` can be used as the `--test_config` for fabric tests

`python3 main.py -f path/to/.textproto [options]`
