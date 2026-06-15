// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ── Configuration ──────────────────────────────────────────────────

const DATA_BASE_PATH = "../data";
const KNOWN_ARCHITECTURES = ["blackhole", "wormhole_b0"];
const GITHUB_BASE =
    "https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/data_movement";

const TESTS = [
    {
        name: "One from One Packet Sizes 2.0",
        csv: "One from One Packet Sizes 2.0.csv",
        tags: ["read", "unicast", "L1", "single_core", "2.0_api"],
        directory: "one_from_one",
    },
    {
        name: "One to One Packet Sizes",
        csv: "One to One Packet Sizes.csv",
        tags: ["write", "unicast", "L1", "single_core"],
        directory: "one_to_one",
    },
    {
        name: "One from All Packet Sizes",
        csv: "One from All Packet Sizes.csv",
        tags: ["read", "unicast", "L1", "multi_core"],
        directory: "one_from_all",
    },
];

const INTERNAL_COLUMNS = new Set([
    "Run Host ID",
    "Architecture",
    "RISC-V Processor",
]);

const METRIC_COLUMNS = new Set([
    "Latency (cycles)",
    "Bandwidth (bytes/cycle)",
    "Combined Bandwidth (bytes/cycle)",
    "Total Bytes",
]);

const DEFAULT_X_AXIS = "Transaction Size (bytes)";
const DEFAULT_SERIES = "Number of Transactions";

const SERIES_COLORS = [
    "#7b68ee",
    "#ff6b6b",
    "#4ecdc4",
    "#ffa726",
    "#66bb6a",
    "#ab47bc",
    "#29b6f6",
    "#ef5350",
    "#26a69a",
    "#ffc107",
];

// ── State ──────────────────────────────────────────────────────────

let state = {
    selectedTest: null,
    selectedArch: null,
    availableArchs: [],
    activeTags: new Set(),
    searchQuery: "",
    yAxisMode: "bandwidth",
    visibleSeries: new Set(),
    csvData: null,
    analysis: null,
    chartConfig: null,
};

// ── Column Analysis ────────────────────────────────────────────────

function analyzeColumns(rows) {
    if (!rows.length) return null;

    const allColumns = Object.keys(rows[0]);
    const metrics = [];
    const sweepDimensions = [];
    const constants = {};

    for (const col of allColumns) {
        const values = [...new Set(rows.map((r) => r[col]))];
        const isConstant = values.length <= 1;

        if (METRIC_COLUMNS.has(col)) {
            metrics.push(col);
        } else if (INTERNAL_COLUMNS.has(col)) {
            if (!isConstant) {
                sweepDimensions.push({ column: col, values });
            }
        } else if (isConstant) {
            constants[col] = values[0];
        } else {
            sweepDimensions.push({ column: col, values });
        }
    }

    return { metrics, sweepDimensions, constants };
}

function buildChartConfig(analysis) {
    const { metrics, sweepDimensions } = analysis;

    const bandwidthCol =
        metrics.find((m) => m.includes("Bandwidth")) || null;
    const latencyCol = metrics.find((m) => m.includes("Latency")) || null;

    const xAxisDim = sweepDimensions.find(
        (d) => d.column === DEFAULT_X_AXIS
    );
    const xAxis = xAxisDim ? DEFAULT_X_AXIS : sweepDimensions[0]?.column;

    const seriesDim = sweepDimensions.find(
        (d) => d.column === DEFAULT_SERIES && d.column !== xAxis
    );
    const seriesGrouper = seriesDim ? DEFAULT_SERIES : null;

    return { xAxis, seriesGrouper, bandwidthCol, latencyCol };
}

// ── CSV Loading ────────────────────────────────────────────────────

function loadCSV(url) {
    return new Promise((resolve, reject) => {
        Papa.parse(url, {
            download: true,
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => resolve(results.data),
            error: (err) => reject(err),
        });
    });
}

async function probeArchitectures(test) {
    const available = [];
    const probes = KNOWN_ARCHITECTURES.map(async (arch) => {
        const url = `${DATA_BASE_PATH}/${arch}/${test.csv}`;
        try {
            const resp = await fetch(url, { method: "HEAD" });
            if (resp.ok) available.push(arch);
        } catch {
            // not available
        }
    });
    await Promise.all(probes);
    available.sort();
    return available;
}

// ── Tag Collection ─────────────────────────────────────────────────

function getAllTags() {
    const tags = new Set();
    for (const test of TESTS) {
        for (const tag of test.tags) {
            tags.add(tag);
        }
    }
    return [...tags].sort();
}

// ── Filtering ──────────────────────────────────────────────────────

function getFilteredTests() {
    return TESTS.filter((test) => {
        if (
            state.searchQuery &&
            !test.name.toLowerCase().includes(state.searchQuery.toLowerCase())
        ) {
            return false;
        }
        if (state.activeTags.size > 0) {
            for (const tag of state.activeTags) {
                if (!test.tags.includes(tag)) return false;
            }
        }
        return true;
    });
}

// ── Chart Rendering (Plotly) ───────────────────────────────────────

function formatBytes(value) {
    if (value >= 1024) {
        return value / 1024 + "K";
    }
    return String(value);
}

function getPlotData(rows, config) {
    const yCol =
        state.yAxisMode === "bandwidth"
            ? config.bandwidthCol
            : config.latencyCol;

    if (!yCol) return new Map();

    const series = new Map();
    for (const row of rows) {
        const seriesKey = config.seriesGrouper
            ? String(row[config.seriesGrouper])
            : "all";

        if (!state.visibleSeries.has(seriesKey)) continue;

        if (!series.has(seriesKey)) {
            series.set(seriesKey, { x: [], y: [] });
        }
        const s = series.get(seriesKey);
        s.x.push(row[config.xAxis]);
        s.y.push(row[yCol]);
    }

    for (const s of series.values()) {
        const indices = s.x.map((_, i) => i).sort((a, b) => s.x[a] - s.x[b]);
        s.x = indices.map((i) => s.x[i]);
        s.y = indices.map((i) => s.y[i]);
    }

    return series;
}

function renderChart(rows, config) {
    const plotData = getPlotData(rows, config);

    const traces = [];
    let colorIdx = 0;
    const sortedKeys = [...plotData.keys()].sort(
        (a, b) => Number(a) - Number(b)
    );

    for (const key of sortedKeys) {
        const s = plotData.get(key);
        const color = SERIES_COLORS[colorIdx % SERIES_COLORS.length];
        traces.push({
            x: s.x,
            y: s.y,
            type: "scatter",
            mode: "lines+markers",
            name: config.seriesGrouper ? String(key) : "Data",
            line: { color, width: 2 },
            marker: { color, size: 5 },
            hovertemplate:
                `<b>${config.xAxis}:</b> %{x}<br>` +
                `<b>%{fullData.name}:</b> %{y:.4f}<extra></extra>`,
        });
        colorIdx++;
    }

    const isLatency = state.yAxisMode === "latency";
    const yLabel = isLatency
        ? "Latency (cycles)"
        : "Bandwidth (bytes/cycle)";

    const layout = {
        xaxis: {
            type: "log",
            dtick: Math.log10(2),
            title: { text: config.xAxis, font: { color: "#a0a0b0" } },
            tickfont: { color: "#a0a0b0" },
            tickformat: "",
            tickvals: getBase2Ticks(traces),
            ticktext: getBase2Ticks(traces).map(formatBytes),
            gridcolor: "#2a2a4a",
            zerolinecolor: "#2a2a4a",
            linecolor: "#2a2a4a",
        },
        yaxis: {
            type: isLatency ? "log" : "linear",
            title: { text: yLabel, font: { color: "#a0a0b0" } },
            tickfont: { color: "#a0a0b0" },
            gridcolor: "#2a2a4a",
            zerolinecolor: "#2a2a4a",
            linecolor: "#2a2a4a",
            rangemode: isLatency ? "normal" : "tozero",
        },
        plot_bgcolor: "#16213e",
        paper_bgcolor: "#16213e",
        font: { color: "#e0e0e0" },
        legend: {
            font: { color: "#e0e0e0" },
            bgcolor: "rgba(0,0,0,0)",
        },
        margin: { t: 30, r: 30, b: 60, l: 70 },
        hovermode: "x unified",
    };

    const plotConfig = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        displaylogo: false,
    };

    Plotly.newPlot("chart", traces, layout, plotConfig);
}

function getBase2Ticks(traces) {
    let min = Infinity;
    let max = -Infinity;
    for (const t of traces) {
        for (const v of t.x) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }
    if (!isFinite(min)) return [];
    const ticks = [];
    let exp = Math.floor(Math.log2(min));
    while (Math.pow(2, exp) <= max) {
        const val = Math.pow(2, exp);
        if (val >= min) ticks.push(val);
        exp++;
    }
    return ticks;
}

// ── UI Rendering ───────────────────────────────────────────────────

function renderTagChips() {
    const container = document.getElementById("tag-chips");
    container.innerHTML = "";
    for (const tag of getAllTags()) {
        const chip = document.createElement("span");
        chip.className =
            "tag-chip" + (state.activeTags.has(tag) ? " active" : "");
        chip.textContent = tag;
        chip.addEventListener("click", () => {
            if (state.activeTags.has(tag)) {
                state.activeTags.delete(tag);
            } else {
                state.activeTags.add(tag);
            }
            renderTagChips();
            renderTestList();
        });
        container.appendChild(chip);
    }
}

function renderTestList() {
    const list = document.getElementById("test-list");
    list.innerHTML = "";
    const filtered = getFilteredTests();

    for (const test of filtered) {
        const li = document.createElement("li");
        if (state.selectedTest && state.selectedTest.name === test.name) {
            li.className = "selected";
        }

        const nameSpan = document.createElement("span");
        nameSpan.textContent = test.name;
        li.appendChild(nameSpan);

        const tagsDiv = document.createElement("div");
        tagsDiv.className = "test-tags";
        for (const tag of test.tags) {
            const tagSpan = document.createElement("span");
            tagSpan.className = "test-tag";
            tagSpan.textContent = tag;
            tagsDiv.appendChild(tagSpan);
        }
        li.appendChild(tagsDiv);

        li.addEventListener("click", () => selectTest(test));
        list.appendChild(li);
    }
}

function renderArchChips() {
    const container = document.getElementById("arch-chips");
    container.innerHTML = "";

    for (const arch of state.availableArchs) {
        const chip = document.createElement("span");
        chip.className =
            "arch-chip" + (state.selectedArch === arch ? " active" : "");
        chip.textContent = arch;
        chip.addEventListener("click", () => {
            if (state.selectedArch !== arch) {
                state.selectedArch = arch;
                loadAndRender();
            }
        });
        container.appendChild(chip);
    }
}

function renderTestInfo(analysis) {
    const container = document.getElementById("test-info");
    container.innerHTML = "";

    if (!analysis) return;

    for (const [key, value] of Object.entries(analysis.constants)) {
        const item = document.createElement("span");
        item.className = "info-item";
        item.innerHTML = `<span class="info-label">${key}:</span> <span class="info-value">${value}</span>`;
        container.appendChild(item);
    }
}

function renderSeriesCheckboxes(config, rows) {
    const container = document.getElementById("series-checkboxes");
    container.innerHTML = "";

    if (!config.seriesGrouper) return;

    const values = [
        ...new Set(rows.map((r) => r[config.seriesGrouper])),
    ].sort((a, b) => Number(a) - Number(b));

    state.visibleSeries = new Set(values.map(String));

    values.forEach((val, idx) => {
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.addEventListener("change", () => {
            const key = String(val);
            if (checkbox.checked) {
                state.visibleSeries.add(key);
            } else {
                state.visibleSeries.delete(key);
            }
            renderChart(state.csvData, state.chartConfig);
        });

        const dot = document.createElement("span");
        dot.className = "series-color-dot";
        dot.style.backgroundColor =
            SERIES_COLORS[idx % SERIES_COLORS.length];

        label.appendChild(checkbox);
        label.appendChild(dot);
        label.appendChild(document.createTextNode(" " + val));
        container.appendChild(label);
    });
}

// ── Main Actions ───────────────────────────────────────────────────

async function selectTest(test) {
    state.selectedTest = test;

    document.getElementById("empty-state").style.display = "none";
    document.getElementById("test-view").style.display = "block";

    document.getElementById("test-name").textContent = test.name;
    document.getElementById("readme-link").href =
        `${GITHUB_BASE}/${test.directory}/README.md`;

    renderTestList();

    state.availableArchs = await probeArchitectures(test);
    if (state.availableArchs.length === 0) {
        document.getElementById("test-info").innerHTML =
            '<span class="info-item"><span class="info-value">No CSV data found for any architecture.</span></span>';
        document.getElementById("arch-chips").innerHTML = "";
        document.getElementById("series-checkboxes").innerHTML = "";
        Plotly.purge("chart");
        return;
    }

    if (
        !state.selectedArch ||
        !state.availableArchs.includes(state.selectedArch)
    ) {
        state.selectedArch = state.availableArchs[0];
    }
    renderArchChips();
    await loadAndRender();
}

async function loadAndRender() {
    const test = state.selectedTest;
    if (!test || !state.selectedArch) return;

    const url = `${DATA_BASE_PATH}/${state.selectedArch}/${test.csv}`;

    try {
        const rows = await loadCSV(url);
        state.csvData = rows;

        const analysis = analyzeColumns(rows);
        state.analysis = analysis;
        const config = buildChartConfig(analysis);
        state.chartConfig = config;

        renderArchChips();
        renderTestInfo(analysis);
        renderSeriesCheckboxes(config, rows);
        renderChart(rows, config);

        setupYAxisListeners();
    } catch (err) {
        document.getElementById("test-info").innerHTML =
            `<span class="info-item"><span class="info-value">Error loading CSV: ${err.message}</span></span>`;
    }
}

function setupYAxisListeners() {
    const radios = document.querySelectorAll('input[name="y-axis"]');
    const handler = (e) => {
        state.yAxisMode = e.target.value;
        renderChart(state.csvData, state.chartConfig);
    };
    for (const radio of radios) {
        const newRadio = radio.cloneNode(true);
        newRadio.checked = radio.value === state.yAxisMode;
        radio.parentNode.replaceChild(newRadio, radio);
        newRadio.addEventListener("change", handler);
    }
}

// ── Init ───────────────────────────────────────────────────────────

function init() {
    renderTagChips();
    renderTestList();

    document.getElementById("search-input").addEventListener("input", (e) => {
        state.searchQuery = e.target.value;
        renderTestList();
    });
}

document.addEventListener("DOMContentLoaded", init);
