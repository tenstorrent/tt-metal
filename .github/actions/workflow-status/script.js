const core = require("@actions/core");

async function run() {
  try {
    const requiredJobs = core.getInput("required-jobs").split(",").map(j => j.trim());
    const optionalJobs = core.getInput("optional-jobs").split(",").map(j => j.trim()).filter(Boolean);

    const needs = JSON.parse(process.env.NEEDS_CONTEXT);
    console.log("Needs context:", needs);

    // Check required jobs
    for (const job of requiredJobs) {
      const result = needs[job]?.result;
      console.log(`Job: ${job}, Result: ${result}`);
      if (result !== "success") {
        core.setFailed(`Required job '${job}' did not succeed.`);
      }
    }

    // Check optional jobs (treat skipped as success)
    for (const job of optionalJobs) {
      const result = needs[job]?.result || "success"; // Default to success if missing
      console.log(`Job: ${job}, Result: ${result}`);
      if (result === "failure") {
        core.setFailed(`Optional job '${job}' failed.`);
      }
    }

    console.log("Workflow was successful.");
  } catch (error) {
    core.setFailed(`Error: ${error.message}`);
  }
}

run();
