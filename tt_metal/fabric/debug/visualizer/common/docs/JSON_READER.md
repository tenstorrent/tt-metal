# JSON reader

`fabric_visualizer_json_reader.hpp` is the strict reader the visualizer's stages read each other's JSON files with,
such as the manifest. A reader built on it fails when the file and the reader disagree about which keys there are,
whatever their values, and names the key by its JSON pointer.

The model to keep in mind is a **checklist**. Every object you read starts with all its keys unticked. Reading a key
ticks it. When you're done with the object, any key still unticked is an error. Everything else in the library is
either how you build that checklist or how a failure says where it happened.

## The two types

JSON has six kinds of value: object, array, string, number, bool and null. The reader has two classes for them.

### `JsonValueReader`: a value you're holding, and where it is

A small handle: a reference to one JSON value and that value's pointer, such as `/meshes/M0/chips/C1`. It keeps no
state. You use it to turn the value into a C++ type:

| JSON value | Call | C++ result |
| --- | --- | --- |
| string | `read<std::string>()` | `std::string` |
| number | `read<uint32_t>()`, `read<uint64_t>()`, `read<int>()`… | that integer type, checked to fit |
| bool | `read<bool>()` | `bool` |
| null or a value | `read_nullable<T>()` | `std::optional<T>` |
| object | `read_object(fn)` | whatever `fn` returns |
| null or an object | `read_nullable_object(fn)` | `std::optional` of what `fn` returns |
| array | `read_array(fn)`, `read_array_of<T>()` | what `fn` builds, or `std::vector<T>` |

A value of another JSON type fails, such as a number written as a string, a fraction read as an integer, or a negative
number read as unsigned.

### `JsonObjectReader`: an object being read, with its checklist

The only part with state: the set of keys ticked so far. Every way of reading a key ticks it:

| Call | Use it when | If the key is missing |
| --- | --- | --- |
| `take(key)`, `take<T>(key)` | the writer always writes the key | fails |
| `take_optional(key)` | the writer leaves the key out where it doesn't apply | returns `nullopt` |
| `skip(key)` | the key must be there, but the reader has no use for it, such as a list derived from the rest of the file | fails |
| `check(key, expected)` | the reader already knows the value, such as the file's version | fails, as does a different value |
| `take_each(fn)` | the keys are ids, such as `M0` and `M1`, so you can't name them ahead of time | visits each key not yet ticked, in the file's order |

`take(key)` returns a `JsonValueReader`, and you're back at the first table deciding what C++ type it becomes.
`take<T>(key)` is `take(key).read<T>()`.

Reading a key twice fails too: it means the reader has a bug.

Where the writer has nothing to write it writes `null`, so a value that can be absent is taken with `take` and read
with `read_nullable`. `take_optional` is only for a key the writer sometimes leaves out altogether.

## Objects are read through a function

You can't make a `JsonObjectReader` yourself. You call `value.read_object(fn)`, which:

1. checks the value is an object;
2. makes the checklist;
3. calls `fn(object_reader)`;
4. once `fn` returns, fails if any key is unticked;
5. returns what `fn` returned.

So the "did I read everything?" check runs at the end of every object, at every level of nesting, and can't be
forgotten. Arrays need no checklist: `read_array` visits every element, so none can be skipped.

To read a whole document, start from its top level, `JsonValueReader(document, "Fabric manifest")`, and call
`read_object` on it. The name you give starts every error.

## An example

The manifest's `run` block, without the build it records:

```json
"run": {
  "run_id": "20261109T120000Z-1a2b3c4d",
  "mpi_rank": 0,
  "world_size": 2,
  "host_name": "wh-lb-85",
  "written_at": "2026-11-09T12:00:00Z"
}
```

read into `RunInfo`:

```cpp
RunInfo run = top.take("run").read_object([](JsonObjectReader& run_json) {
    RunInfo run;
    run.run_id = run_json.take<std::string>("run_id");
    run.mpi_rank = run_json.take<uint32_t>("mpi_rank");
    run.world_size = run_json.take<uint32_t>("world_size");
    // Written only for a job tt-run splits into sub-contexts.
    if (const auto subcontext = run_json.take_optional("subcontext")) {
        run.subcontext = subcontext->read_object([](JsonObjectReader& subcontext_json) { ... });
    }
    run.host_name = run_json.take<std::string>("host_name");
    run.written_at = run_json.take<std::string>("written_at");
    return run;
});  // the checklist for /run is checked here
```

The reader mirrors the writer: each JSON object is one `read_object` returning one model struct, and each key is one
line filling one member.

## When the writer and reader drift

If the writer gains a key the reader doesn't read, whatever its value, the object's checklist fails when its function
returns:

```
Fabric manifest: /run has keys this reader doesn't read: /run/pid
```

If the reader expects a key the writer no longer writes, `take` fails right there:

```
Fabric manifest: /run/host_name is missing
```

You never build those pointers. Each reader extends its parent's pointer as it goes down (`/run`, then
`/run/host_name`), so every error names exactly where it happened.

## What it doesn't check

- **What a value means beyond its JSON type.** The reader knows `"fabric_2d"` is a string, not that it names a
  topology. The code reading it converts it, and when it can't, calls `value.fail("...")`, which names the value by its
  pointer the same way:

  ```
  Fabric manifest: /meshes/M0/chips/C0/routers/E0/link/peer names M0/C9/W0, a chip this manifest doesn't have
  ```

- **Whether values come back right.** The reader catches keys that are missing or extra. Wrong values take a round
  trip: write, read back, write again, and compare the two documents.

In short: `JsonValueReader` answers "what is this value as a C++ type?", `JsonObjectReader` answers "which keys have I
read?", and `read_object` joins them and makes the check at the end automatic.
