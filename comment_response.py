Comment = r"""

So right now we have a model that is composed of layers, each layer has it's own forward function: it boils down to this in principle:
class model
def forward(metadata_arg):
for i in range(num_layer):
layers[i].forward(metadata_arg)

class layer:
def forward (metadata_arg):
ttnn.matmul()
ttnn.add()
ttnn.op_with_metadata_arg(metadata_arg)

metadata_arg is noted here as an example of a runtime arg that need trace patching

and then in the path that calls model, we do:
model.forward(metadata_arg)

Rignt now (traced runtime args aside), we do:
model.forward() // compile pass
ttnn.begin_trace_capture()
model.forward() // capture trace
trace_id = ttnn.end_trace_capture()

for i in range (num_model_invocations):
ttnn.execute_trace(trace_id)

how would the trace capture look like under new APIs?
would we have to add builder.build/add to all of our ops ?
there are also cases where we want to run model untraced, for debug purposes, current trace API
allows us to do so by simply not calling trace capture and just calling model_fwd() ->
are we going to be able to do the same here?
"""
