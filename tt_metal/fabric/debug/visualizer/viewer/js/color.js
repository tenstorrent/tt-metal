const COOL = [93, 226, 194];
const HOT = [255, 114, 114];
const GREY = "#82939c";

function hex(parts) {
  return `#${parts.map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
}

export function stallStroke(score) {
  if (score === null || score === undefined || Number.isNaN(Number(score))) {
    return GREY;
  }
  const t = Math.min(1, Math.max(0, Number(score)));
  return hex(COOL.map((channel, index) => channel + (HOT[index] - channel) * t));
}

export function linkAppearance(link) {
  const status = link.status || link.srcRouter?.capture?.status || "unknown";
  const missing = ["not_captured", "unknown", "unreadable", "unsupported"].includes(status);
  return {
    status,
    missing,
    torn: status === "torn" || Boolean(link.srcRouter?.capture?.torn),
    reset: status === "reset" || link.srcRouter?.capture?.status === "reset",
    intermesh: link.link_class === "intermesh",
    crossHost: Boolean(link.cross_host),
    wrap: Boolean(link.wrap),
    stroke: missing ? GREY : stallStroke(link.stall_score),
  };
}

export function captureAppearance(status) {
  const value = status || "unknown";
  return {
    status: value,
    fill: {
      ok: "#12353a",
      torn: "#3a3118",
      reset: "#3a1818",
      unreadable: "#3a1818",
      not_captured: "#1b2429",
      unknown: "#1b2429",
      unsupported: "#1b2429",
    }[value] || "#1b2429",
    stroke: {
      ok: "#376474",
      torn: "#f5c451",
      reset: "#ff7272",
      unreadable: "#ff7272",
      not_captured: "#82939c",
      unknown: "#82939c",
      unsupported: "#82939c",
    }[value] || "#82939c",
    hatch: value === "torn",
    outline: value === "reset",
  };
}
