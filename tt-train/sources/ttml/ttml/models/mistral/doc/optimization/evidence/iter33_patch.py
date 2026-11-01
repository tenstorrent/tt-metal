"""Iteration 33: the TP dx all-reduce of a fused column-parallel LoRA layer overlapped with its LoRA gradient products.
A second mesh command queue; `all_reduce_async_begin` (reduce-scatter + all-gather on queue 1, pre-allocated buffers
and dedicated semaphores, mesh events against queue 0) / `all_reduce_async_end`; the fused backward issues it right
after `transpose_split` and waits after the products; the column-parallel fused path skips the `broadcast` whose
backward did the all-reduce. TTML_NUM_CQS=1 / TTML_ASYNC_COLLECTIVES=0 fall back to the synchronous path."""
import sys

T = "tt-train/sources/ttml"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:90]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


# ---- mesh: two command queues (env knob)
sub(
    f"{T}/core/mesh_device.cpp",
    "        /* num_command_queues=*/1,\n",
    "        // Two command queues (iteration 33): the TP collectives of the backward run on queue 1 under the LoRA gradient\n"
    "        // products; TTML_NUM_CQS=1 restores the single queue.\n"
    '        static_cast<uint8_t>(std::getenv("TTML_NUM_CQS") != nullptr ? std::strtoul(std::getenv("TTML_NUM_CQS"), nullptr, 10) : 2U),\n',
)
sub(f"{T}/core/mesh_device.cpp", '#include "mesh_device.hpp"\n', '#include "mesh_device.hpp"\n\n#include <cstdlib>\n')
# ---- context: a second semaphore set for the async queue
sub(
    f"{T}/autograd/auto_context.hpp",
    "    [[nodiscard]] core::distributed::CCLResources& get_ccl_resources();\n",
    "    [[nodiscard]] core::distributed::CCLResources& get_ccl_resources();\n"
    "    // The semaphore set of the collectives issued on command queue 1 (iteration 33): disjoint from the queue-0 set\n"
    "    // so a queue-1 collective in flight and the next queue-0 one never share a semaphore.\n"
    "    [[nodiscard]] core::distributed::CCLResources& get_ccl_resources_async();\n",
)
sub(
    f"{T}/autograd/auto_context.hpp",
    "    std::unique_ptr<core::distributed::CCLResources> m_ccl_resources;\n",
    "    std::unique_ptr<core::distributed::CCLResources> m_ccl_resources;\n    std::unique_ptr<core::distributed::CCLResources> m_ccl_resources_async;\n",
)
sub(
    f"{T}/autograd/auto_context.cpp",
    "    m_ccl_resources.reset();\n    m_device_constants.clear();\n",
    "    m_ccl_resources.reset();\n    m_ccl_resources_async.reset();\n    m_device_constants.clear();\n",
)
sub(
    f"{T}/autograd/auto_context.cpp",
    """[[nodiscard]] core::distributed::CCLResources& AutoContext::get_ccl_resources() {
    if (!m_ccl_resources) {
        m_ccl_resources = std::make_unique<core::distributed::CCLResources>();
    }
    return *m_ccl_resources;
}
""",
    """[[nodiscard]] core::distributed::CCLResources& AutoContext::get_ccl_resources() {
    if (!m_ccl_resources) {
        m_ccl_resources = std::make_unique<core::distributed::CCLResources>();
    }
    return *m_ccl_resources;
}

[[nodiscard]] core::distributed::CCLResources& AutoContext::get_ccl_resources_async() {
    if (!m_ccl_resources_async) {
        m_ccl_resources_async = std::make_unique<core::distributed::CCLResources>();
    }
    return *m_ccl_resources_async;
}
""",
)
# ---- the async all-reduce
sub(
    f"{T}/ttnn_fixed/distributed/ttnn_ops.hpp",
    """ttnn::Tensor mesh_partition(
    const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);
""",
    """ttnn::Tensor mesh_partition(
    const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis = std::nullopt);

/**
 * An all-reduce in flight on mesh command queue 1 (iteration 33): `all_reduce_async_begin` orders queue 1 after the
 * input's producers on queue 0 (a mesh event), issues the reduce-scatter and the all-gather there with buffers this
 * object owns and the queue-1 semaphore set, and records the completion event; `all_reduce_async_end` makes queue 0
 * wait for it, after which `output` is valid. Queue 0 work issued between the two overlaps the collective. With a
 * single command queue (or TTML_ASYNC_COLLECTIVES=0) `begin` runs the synchronous all-reduce and `end` is a no-op.
 */
struct AsyncAllReduce {
    ttnn::Tensor output;                                               // the all-reduced tensor, valid after end
    std::vector<ttnn::Tensor> scratch;                                 // the reduce-scatter's buffers, alive until end
    std::shared_ptr<tt::tt_metal::distributed::MeshEvent> done{};    // recorded on queue 1 after the all-gather
    bool ended{true};
};
AsyncAllReduce all_reduce_async_begin(const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis);
void all_reduce_async_end(AsyncAllReduce& op);
""",
)
sub(
    f"{T}/ttnn_fixed/distributed/ttnn_ops.cpp",
    "ttnn::Tensor mesh_partition(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {",
    """AsyncAllReduce all_reduce_async_begin(const ttnn::Tensor& tensor, const std::optional<uint32_t> cluster_axis) {
    auto& mesh_device = ttml::autograd::ctx().get_device();
    const char* knob = std::getenv("TTML_ASYNC_COLLECTIVES");
    const bool async_enabled = mesh_device.num_hw_cqs() >= 2U && !(knob != nullptr && std::string(knob) == "0");
    if (!async_enabled) {
        return AsyncAllReduce{.output = all_reduce(tensor, cluster_axis), .scratch = {}, .done = nullptr, .ended = true};
    }
    TT_FATAL(tensor.logical_shape().rank() == 4U, "all_reduce_async_begin supports only 4D tensors");
    constexpr int dim = 3;
    auto& cq0 = mesh_device.mesh_command_queue(0);
    auto& cq1 = mesh_device.mesh_command_queue(1);
    auto& ccl_resources = ttml::autograd::ctx().get_ccl_resources_async();
    uint32_t num_links = ttnn::operations::ccl::common::get_num_links(mesh_device, cluster_axis);
    auto topology = get_topology(cluster_axis);

    // Queue 1 starts after the input's producers on queue 0.
    const auto ready = cq0.enqueue_record_event();
    cq1.enqueue_wait_for_event(ready);

    AsyncAllReduce op;
    op.ended = false;
    {
        auto guard = ttnn::core::with_command_queue_id(ttnn::QueueId(1));
        // The reduce-scatter's intermediates and output are allocated here and kept until `end`: an op-internal buffer
        // freed on the host while queue 1 still uses it would be handed to the next queue-0 allocation.
        auto intermediates = ttnn::experimental::reduce_scatter_minimal_async_create_intermediate_buffer(
            tensor, dim, topology, cluster_axis);
        const auto in_shape = tensor.logical_shape();
        const uint32_t ring = cluster_axis.has_value() ? mesh_device.shape()[*cluster_axis] : mesh_device.num_devices();
        auto scattered = ttnn::empty(
            ttnn::Shape{in_shape[0], in_shape[1], in_shape[2], in_shape[3] / ring},
            tensor.dtype(),
            tensor.layout(),
            &mesh_device,
            tensor.memory_config());
        std::vector<ttnn::Tensor> persistent{intermediates[0], scattered, intermediates[1]};
        auto rs = ttnn::experimental::reduce_scatter_minimal_async(
            tensor,
            persistent,
            dim,
            ccl_resources.get_reduce_scatter_semaphores(),
            ccl_resources.get_barrier_semaphore(),
            ccl_num_links(num_links),
            /* memory_config */ std::nullopt,
            /* intermediate_memory_config */ std::nullopt,
            topology,
            /* subdevice_id */ std::nullopt,
            cluster_axis,
            ccl_env("TTML_CCL_RS_CHUNKS_PER_SYNC"),
            ccl_env("TTML_CCL_RS_WORKERS_PER_LINK"),
            ccl_env("TTML_CCL_RS_BUFFERS_PER_CHANNEL"));
        auto gathered = ttnn::empty(in_shape, tensor.dtype(), tensor.layout(), &mesh_device, tensor.memory_config());
        op.output = ttnn::experimental::all_gather_async(
            rs,
            gathered,
            dim,
            ccl_resources.get_all_gather_semaphore(),
            ccl_num_links(num_links),
            /* memory_config */ std::nullopt,
            topology,
            /* subdevice_id */ std::nullopt,
            cluster_axis,
            /* use_optimal_ccl_for_llama */ false,
            /* barrier_semaphore */ ccl_resources.get_barrier_semaphore(),
            /* use_all_gather_async_via_broadcast */ false,
            ccl_env("TTML_CCL_AG_CHUNKS_PER_SYNC"),
            ccl_env("TTML_CCL_AG_WORKERS_PER_LINK"),
            ccl_env("TTML_CCL_AG_BUFFERS_PER_CHANNEL"));
        op.scratch = {std::move(persistent[0]), std::move(persistent[1]), std::move(persistent[2]), rs, gathered};
        op.done = std::make_shared<tt::tt_metal::distributed::MeshEvent>(cq1.enqueue_record_event());
    }
    return op;
}

void all_reduce_async_end(AsyncAllReduce& op) {
    if (op.ended) {
        return;
    }
    auto& mesh_device = ttml::autograd::ctx().get_device();
    mesh_device.mesh_command_queue(0).enqueue_wait_for_event(*op.done);
    op.ended = true;
}

ttnn::Tensor mesh_partition(const ttnn::Tensor& tensor, const int dim, const std::optional<uint32_t> cluster_axis) {""",
)
s = open(f"{T}/ttnn_fixed/distributed/ttnn_ops.cpp").read()
if "#include <cstdlib>" not in s:
    s = s.replace('#include "ttnn_ops.hpp"\n', '#include "ttnn_ops.hpp"\n\n#include <cstdlib>\n#include <string>\n', 1)
if "ttnn/core.hpp" not in s:
    s = s.replace(
        '#include "ttnn_ops.hpp"\n',
        '#include "ttnn_ops.hpp"\n\n#include <tt-metalium/mesh_event.hpp>\n#include <ttnn/core.hpp>\n',
        1,
    )
open(f"{T}/ttnn_fixed/distributed/ttnn_ops.cpp", "w").write(s)
h = open(f"{T}/ttnn_fixed/distributed/ttnn_ops.hpp").read()
if "mesh_event.hpp" not in h:
    h = h.replace(
        "#include <core/ttnn_all_includes.hpp>\n",
        "#include <core/ttnn_all_includes.hpp>\n#include <tt-metalium/mesh_event.hpp>\n\n#include <memory>\n#include <vector>\n",
        1,
    )
open(f"{T}/ttnn_fixed/distributed/ttnn_ops.hpp", "w").write(h)
# ---- binding
sub(
    f"{T}/nanobind/nb_core.cpp",
    """        py_distributed.def(
            "all_reduce",
            &ttml::ttnn_fixed::distributed::all_reduce,
            nb::arg("tensor"),
            nb::arg("cluster_axis") = nb::none(),
            "Raw all_reduce without autograd tracking. Returns a new ttnn::Tensor.");
""",
    """        py_distributed.def(
            "all_reduce",
            &ttml::ttnn_fixed::distributed::all_reduce,
            nb::arg("tensor"),
            nb::arg("cluster_axis") = nb::none(),
            "Raw all_reduce without autograd tracking. Returns a new ttnn::Tensor.");
        nb::class_<ttml::ttnn_fixed::distributed::AsyncAllReduce>(py_distributed, "AsyncAllReduce")
            .def_ro("output", &ttml::ttnn_fixed::distributed::AsyncAllReduce::output)
            .def_ro("ended", &ttml::ttnn_fixed::distributed::AsyncAllReduce::ended);
        py_distributed.def(
            "all_reduce_async_begin",
            &ttml::ttnn_fixed::distributed::all_reduce_async_begin,
            nb::arg("tensor"),
            nb::arg("cluster_axis") = nb::none(),
            "Issue an all_reduce on mesh command queue 1 (iteration 33); `all_reduce_async_end` makes queue 0 wait "
            "for it, then `.output` is valid. Synchronous with one queue or TTML_ASYNC_COLLECTIVES=0.");
        py_distributed.def(
            "all_reduce_async_end",
            &ttml::ttnn_fixed::distributed::all_reduce_async_end,
            nb::arg("op"),
            "Queue 0 waits for the queue-1 all_reduce of `op`.");
""",
)
# ---- python: fused backward issues the dx all-reduce first, the column-parallel fused path drops the broadcast
F = f"{T}/ttml/modules/fused_lora.py"
sub(
    F,
    """        dh_scale,
        reduce_axis,
        h_reduce_axis,
    ):
        cfg = _compute_config()""",
    """        dh_scale,
        reduce_axis,
        h_reduce_axis,
        dx_reduce_axis=None,
    ):
        cfg = _compute_config()""",
)
sub(
    F,
    "        ctx.meta = (scaling, in_features, out_features, rank, dh_scale, reduce_axis, wide)\n",
    "        ctx.meta = (scaling, in_features, out_features, rank, dh_scale, reduce_axis, wide, dx_reduce_axis)\n",
)
sub(
    F,
    "        scaling, in_features, out_features, rank, dh_scale, reduce_axis, wide = ctx.meta\n",
    "        scaling, in_features, out_features, rank, dh_scale, reduce_axis, wide, dx_reduce_axis = ctx.meta\n",
)
sub(
    F,
    """        ttnn.deallocate(dcat_t)
        grad_b = skinny_contract(h, grad_output, scaling, transpose_out=True)  # s * g^T h -> [out, r]
        if reduce_axis is not None:
            s_dh = ttml.core.distributed.all_reduce(s_dh, reduce_axis)
        grad_a = skinny_contract(s_dh, x_val)  # (s dh)^T x -> [r, in] (+ the tail columns when x is wide)
        if wide:
            grad_a = ttnn.slice(grad_a, [0, 0, 0, 0], [1, 1, rank, in_features], [1, 1, 1, 1])
        return dx, None, grad_a, grad_b
""",
    """        ttnn.deallocate(dcat_t)
        # Column-parallel layer (iteration 33): the TP all-reduce of dx (the input's broadcast backward, done here
        # instead) runs on command queue 1 while the LoRA gradient products run on queue 0.
        dx_reduce = ttml.core.distributed.all_reduce_async_begin(dx, dx_reduce_axis) if dx_reduce_axis is not None else None
        grad_b = skinny_contract(h, grad_output, scaling, transpose_out=True)  # s * g^T h -> [out, r]
        if reduce_axis is not None:
            s_dh = ttml.core.distributed.all_reduce(s_dh, reduce_axis)
        grad_a = skinny_contract(s_dh, x_val)  # (s dh)^T x -> [r, in] (+ the tail columns when x is wide)
        if wide:
            grad_a = ttnn.slice(grad_a, [0, 0, 0, 0], [1, 1, rank, in_features], [1, 1, 1, 1])
        if dx_reduce is not None:
            ttml.core.distributed.all_reduce_async_end(dx_reduce)
            dx = dx_reduce.output
        return dx, None, grad_a, grad_b
""",
)
sub(
    F,
    """def fused_lora_linear_t(
    x, weight_cat, lora_A, lora_B, scaling, shape, tail_scale=1.0, dh_scale=1.0, reduce_axis=None, h_reduce_axis=None
):
    in_features, out_features = shape
    return FusedLoraLinearT.apply(
        x,
        weight_cat,
        lora_A,
        lora_B,
        scaling,
        in_features,
        out_features,
        tail_scale,
        dh_scale,
        reduce_axis,
        h_reduce_axis,
    )""",
    """def fused_lora_linear_t(
    x,
    weight_cat,
    lora_A,
    lora_B,
    scaling,
    shape,
    tail_scale=1.0,
    dh_scale=1.0,
    reduce_axis=None,
    h_reduce_axis=None,
    dx_reduce_axis=None,
):
    in_features, out_features = shape
    return FusedLoraLinearT.apply(
        x,
        weight_cat,
        lora_A,
        lora_B,
        scaling,
        in_features,
        out_features,
        tail_scale,
        dh_scale,
        reduce_axis,
        h_reduce_axis,
        dx_reduce_axis,
    )""",
)
L = f"{T}/ttml/modules/lora.py"
s = open(L).read()
a = s.index("class LoraColumnParallelLinear")
b = s.index("    def forward(self, x):", a)
c = s.index("        x = ttml.ops.distributed.broadcast(x, self.cluster_axis)\n", b)
d = s.index("            return fused_lora_linear_t(", c)
e = s.index(")\n", d)
seg = s[c : e + 2]
print("column-parallel forward segment:\n" + seg)
